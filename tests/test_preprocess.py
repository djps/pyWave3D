"""The preprocessors of other solvers' output, against the planes of a pyWave3D run.

A nonlinear focused run is recorded plane by plane. The same planes are then given to each
preprocessor, as a frequency-domain solver would give them (any axis order and phase
convention), and as time series of the kind k-wave-python and fullwave25 record (sampled at a
time step that does not divide the period, after a transient, with drift for FDTD). Each must
give back the metrics of the run.
"""

from __future__ import annotations

import logging
import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import pywave3d as pw
from conftest import focused_model
from pywave3d.options import Options
from pywave3d.preprocess import HarmonicFit, fdtd, spectral, time_domain

F0 = 1e6


def assert_metrics_equal(a: Any, b: Any, rtol: float, path: str = "") -> None:
    """Every number of two metrics dicts agrees to ``rtol`` (relative to the largest of its kind)."""
    if isinstance(a, dict):
        assert set(a) == set(b), (path, set(a) ^ set(b))
        for k in a:
            assert_metrics_equal(a[k], b[k], rtol, f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (u, v) in enumerate(zip(a, b, strict=True)):
            assert_metrics_equal(u, v, rtol, f"{path}[{i}]")
    elif isinstance(a, (bool, str)) or a is None:
        assert a == b, (path, a, b)
    else:
        a, b = float(a), float(b)
        if math.isnan(a) or math.isnan(b):
            assert math.isnan(a) and math.isnan(b), (path, a, b)
        else:
            assert b == pytest.approx(a, rel=rtol, abs=rtol * 1e-3), (path, a, b)


@pytest.fixture(scope="module")
def run() -> tuple[pw.WaveSolver, pw.Results, np.ndarray, np.ndarray]:
    """A nonlinear run and its every plane: (solver, results, z, planes (nz, K, nx, ny))."""
    model = focused_model(radius=5e-3, depth=20e-3, lx=14e-3, lz=26e-3, amplitude=8e5)
    grid = model.make_grid(Options({"NumberOfPointsPerWaveLength": 4, "SamplingFrequencyMultiplier": 8}))
    solver = pw.WaveSolver(model, grid, pw.SolverSettings(points=[("p.txt", (*grid.centre, 0.015))]))
    planes: list[np.ndarray] = []
    zs: list[float] = []
    add = pw.PlaneRecorder.add

    def spy(self: pw.PlaneRecorder, z: float, harmonics: np.ndarray, n_active: int | None = None) -> None:
        zs.append(z)
        planes.append(harmonics.astype(complex))
        add(self, z, harmonics, n_active)

    pw.PlaneRecorder.add = spy  # type: ignore[method-assign]
    try:
        res = solver.run()
    finally:
        pw.PlaneRecorder.add = add  # type: ignore[method-assign]
    return solver, res, np.array(zs), np.array(planes)


def common(solver: pw.WaveSolver, res: pw.Results) -> dict[str, Any]:
    medium = solver.model.medium
    return {
        "frequency": F0,
        "sound_speed": medium.sound_speed,
        "density": medium.density,
        "settings": solver.record_settings(),
        "alpha": res.alpha,
        "focusing_depth": res.focusing_depth,
    }


def without_run_info(m: dict[str, Any]) -> dict[str, Any]:
    m = dict(m)
    m.pop("godunov_from_z")  # a property of the march, not of the field
    return m


def test_spectral_planes_give_the_run_back(run: Any) -> None:
    solver, res, z, planes = run
    g = res.grid
    out = spectral.from_harmonics(planes, x=g.x, y=g.y, z=z, **common(solver, res))
    assert_metrics_equal(without_run_info(res.metrics()), without_run_info(out.metrics()), 1e-12)
    np.testing.assert_array_equal(out.points["p.txt"][1], res.points["p.txt"][1])


def test_spectral_layouts_and_conventions(run: Any) -> None:
    """Axes in another order, absolute phase, exp(-i omega t): the same field."""
    solver, res, z, planes = run
    g = res.grid
    k = 2 * np.pi * F0 / solver.model.medium.sound_speed
    n = np.arange(1, planes.shape[1] + 1)
    absolute = planes * np.exp(-1j * n[None, :, None, None] * k * (z - z[0])[:, None, None, None])
    xyzk = np.transpose(absolute.conj(), (2, 3, 0, 1))
    out = spectral.from_harmonics(
        xyzk, layout="xyzk", x=g.x, y=g.y, z=z, phase="absolute", time_sign=-1, **common(solver, res)
    )
    assert_metrics_equal(without_run_info(res.metrics()), without_run_info(out.metrics()), 1e-9)


def test_spectral_single_harmonic_2d(run: Any) -> None:
    """A CW field (no harmonic axis) in 2D (no y axis): the xz plane of the run's fundamental."""
    solver, res, z, planes = run
    g = res.grid
    _, iy = g.centre_index
    xz = planes[:, 0, :, iy]  # (nz, nx)
    out = spectral.from_harmonics(xz, layout="zx", x=g.x, z=z, **common(solver, res) | {"alpha": None})
    assert out.grid.ny == 1 and out.grid.dy == 1.0
    np.testing.assert_allclose(out.axis[:, 0], planes[:, 0, g.centre_index[0], iy])
    m = out.metrics(level_db=-6)
    assert math.isnan(m["focal_width_y_6dB"]) and m["focal_width_x_6dB"] > 0
    # power per metre in y: the line integral of the intensity
    i_line = np.abs(xz[0]) ** 2 / (2 * solver.model.medium.impedance)
    assert m["power_source"] == pytest.approx(i_line.sum() * g.dx)


def synthesize(planes: np.ndarray, z: np.ndarray, dt: float, nt: int, c0: float, t0: float = 0.0) -> np.ndarray:
    """Time series p(x, y, z, t) of retarded harmonics (nz, K, nx, ny) at t = t0 + i dt.

    Before the wave reaches a plane the series is noise, so only a late window is periodic.
    """
    t = t0 + dt * np.arange(nt)
    n = np.arange(1, planes.shape[1] + 1)
    w = 2 * np.pi * F0
    phase = np.exp(1j * w * n[None, :, None] * (t[None, None, :] - (z - z[0])[:, None, None] / c0))  # (nz, K, nt)
    p = np.einsum("zkxy,zkt->xyzt", planes, phase).real
    rng = np.random.default_rng(3)
    arrival = (z - z[0]) / c0 + 2 / F0  # (nz,)
    late = t[None, :] >= arrival[:, None] + t0  # (nz, nt)
    noise = rng.standard_normal(p.shape) * np.abs(planes[:, 0]).max()
    return np.where(late[None, None], p, noise)


@pytest.fixture(scope="module")
def coarse(run: Any) -> tuple[Any, pw.Results, np.ndarray, np.ndarray, pw.Results]:
    """Every third plane, and the results of those planes given as harmonics."""
    solver, res, z, planes = run
    zc, pc = z[::3], planes[::3]
    g = res.grid
    ref = spectral.from_harmonics(pc, x=g.x, y=g.y, z=zc, **common(solver, res))
    return solver, res, zc, pc, ref


def test_time_series_give_the_planes_back(coarse: Any) -> None:
    solver, res, z, planes, ref = coarse
    g, c0 = res.grid, solver.model.medium.sound_speed
    dt = 1 / (F0 * 23.7)  # a period is not a whole number of steps
    duration = (z[-1] - z[0]) / c0 + 5 / F0
    p = synthesize(planes, z, dt, int(duration / dt), c0)
    out = time_domain.from_time_series(
        p,
        dt,
        coords={"x": g.x, "y": g.y, "z": z},
        layout="xyzt",
        propagation="z",
        n_harmonics=planes.shape[1],
        periods=2.0,
        **common(solver, res),
    )
    assert_metrics_equal(without_run_info(ref.metrics()), without_run_info(out.metrics()), 1e-8)


def test_kwave_sensor_data(coarse: Any) -> None:
    """k-wave-python layout: (n_sensor, Nt) in C order of a block mask, propagation along x."""
    solver, res, z, planes, ref = coarse
    g, c0 = res.grid, solver.model.medium.sound_speed
    dt = 1 / (F0 * 19.3)
    start = 40  # record_start_index (1-based)
    nt = int(((z[-1] - z[0]) / c0 + 5 / F0) / dt)
    t0 = (start - 1) * dt
    p = synthesize(planes, z, dt, nt, c0, t0=t0)  # (x, y, z, t) of the field
    beam = np.transpose(p, (2, 0, 1, 3))  # k-Wave grid axes (x = propagation, y, z = transverse)
    # the sensor block sits inside a larger k-Wave grid
    pad = (2, 3, 1)
    shape = tuple(n + 2 * q for n, q in zip(beam.shape[:3], pad, strict=True))
    mask = np.zeros(shape, dtype=bool)
    block = tuple(slice(q, q + n) for q, n in zip(pad, beam.shape[:3], strict=True))
    mask[block] = True
    vec = [
        z[0] - pad[0] * (z[1] - z[0]) + (z[1] - z[0]) * np.arange(shape[0]),
        g.x[0] - pad[1] * g.dx + g.dx * np.arange(shape[1]),
        g.y[0] - pad[2] * g.dy + g.dy * np.arange(shape[2]),
    ]
    kgrid = SimpleNamespace(dim=3, x_vec=vec[0][:, None], y_vec=vec[1][:, None], z_vec=vec[2][:, None], dt=dt)
    data = {"p": beam.reshape(-1, nt)}
    out = time_domain.from_kwave(
        data,
        kgrid,
        sensor_mask=mask,
        record_start_index=start,
        propagation="x",
        n_harmonics=planes.shape[1],
        **common(solver, res),
    )
    assert_metrics_equal(without_run_info(ref.metrics()), without_run_info(out.metrics()), 1e-8)


def test_fullwave_sensor_data(coarse: Any) -> None:
    """fullwave25 layout: sensors by grid index (any order), decimated in time, with PML drift."""
    solver, res, z, planes, ref = coarse
    g, c0 = res.grid, solver.model.medium.sound_speed
    dx = z[1] - z[0]
    modulus = 3
    dt = 1 / (F0 * 31.1)
    nt = int(((z[-1] - z[0]) / c0 + 5 / F0) / (dt * modulus))
    p = synthesize(planes, z, dt * modulus, nt, c0)  # (x, y, z, t)
    beam = np.transpose(p, (2, 0, 1, 3))  # fullwave axes: x axial, y lateral, z elevational
    t = dt * modulus * np.arange(nt)
    beam = beam + 3e4 + 2e10 * t  # slow drift from the absorbing layers (about 1% over the window)
    lo = np.array([7, 2, 5])
    idx = np.argwhere(np.ones(beam.shape[:3], dtype=bool)) + lo
    order = np.random.default_rng(0).permutation(len(idx))
    sensor = SimpleNamespace(outcoords=idx[order], sampling_modulus_time=modulus, is_sparse_grid=False)
    fgrid = SimpleNamespace(dx=dx, dy=g.dx, dz=g.dy, dt=dt, f0=F0, c0=c0)
    origin = (z[0] - lo[0] * dx, g.x[0] - lo[1] * g.dx, g.y[0] - lo[2] * g.dy)
    data = beam.reshape(-1, nt)[order]
    cm = common(solver, res)
    out = fdtd.from_fullwave(
        data,
        sensor,
        fgrid,
        density=cm.pop("density"),
        origin=origin,
        n_harmonics=planes.shape[1],
        **{k: v for k, v in cm.items() if k not in ("frequency", "sound_speed")},
    )
    assert_metrics_equal(without_run_info(ref.metrics()), without_run_info(out.metrics()), 1e-8)
    # without removing the drift, the fit leaves part of it in the harmonics
    cm = common(solver, res)
    bad = fdtd.from_fullwave(
        data,
        sensor,
        fgrid,
        density=cm.pop("density"),
        origin=origin,
        n_harmonics=planes.shape[1],
        detrend=False,
        **{k: v for k, v in cm.items() if k not in ("frequency", "sound_speed")},
    )
    assert abs(bad.metrics()["focus"]["p_positive"] / ref.metrics()["focus"]["p_positive"] - 1) > 1e-3


def test_fdtd_2d(run: Any) -> None:
    """A 2D FDTD record (x axial, y lateral) against the same field given as harmonics."""
    solver, res, z, planes = run
    g, c0 = res.grid, solver.model.medium.sound_speed
    _, iy = g.centre_index
    zc, line = z[::3], planes[::3, :, :, iy]  # (nz, K, nx)
    dt = 1 / (F0 * 27.9)
    nt = int(((zc[-1] - zc[0]) / c0 + 5 / F0) / dt)
    p = synthesize(line[..., None], zc, dt, nt, c0)[:, 0]  # (x_lateral, z, t)
    cm = common(solver, res)
    ref = spectral.from_harmonics(line, layout="zkx", x=g.x, z=zc, **cm)
    out = fdtd.from_time_series(
        np.transpose(p, (1, 0, 2)),
        dt,
        coords={"x": zc, "y": g.x},
        layout="xyt",
        propagation="x",
        n_harmonics=line.shape[1],
        **cm,
    )
    assert out.grid.ny == 1
    assert_metrics_equal(without_run_info(ref.metrics()), without_run_info(out.metrics()), 1e-8)


def test_harmonic_fit() -> None:
    dt = 1 / (F0 * 10.3)
    with pytest.raises(ValueError, match="Nyquist"):
        HarmonicFit(500, dt, F0, 6)
    with pytest.raises(ValueError, match="too short"):
        HarmonicFit(10, dt, F0, 2, periods=2.0)
    fit = HarmonicFit(500, dt, F0, 3, periods=3.0, start_time=1e-7)
    t = 1e-7 + dt * np.arange(500)
    p = np.real((2 - 1j) * np.exp(2j * np.pi * F0 * t)) + np.real(0.5j * np.exp(6j * np.pi * F0 * t))
    h, rel = fit(p)
    np.testing.assert_allclose(h, [2 - 1j, 0, 0.5j], atol=1e-12)
    assert rel < 1e-12


def test_not_steady_is_reported(caplog: pytest.LogCaptureFixture) -> None:
    dt = 1 / (F0 * 12.1)
    t = dt * np.arange(400)
    growing = (t / t[-1])[None, :] * np.sin(2 * np.pi * F0 * t)[None, :] * np.ones((3, 4, 1, 1))  # x, y, z, t
    with caplog.at_level(logging.WARNING, logger="pywave3d.preprocess"):
        time_domain.from_time_series(
            growing,
            dt,
            coords={"x": [0.0, 1e-3, 2e-3], "y": [0, 1e-3, 2e-3, 3e-3], "z": [0.0]},
            frequency=F0,
            sound_speed=1500.0,
            density=1000.0,
            n_harmonics=2,
            periods=4.0,
        )
    assert "not a steady periodic field" in caplog.text


def test_layout_and_sensor_errors() -> None:
    with pytest.raises(ValueError, match="layout"):
        spectral.standard_layout(np.zeros((2, 3, 4)), "zkxy")
    with pytest.raises(ValueError, match="layout"):
        spectral.standard_layout(np.zeros((2, 3)), "kx")
    with pytest.raises(ValueError, match="uniformly"):
        spectral.from_harmonics(
            np.zeros((2, 3)), layout="zx", x=[0, 1, 3], z=[0, 1], frequency=F0, sound_speed=1.0, density=1.0
        )
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1:3] = True
    assert time_domain.sensor_block(mask, (4, 4)) == (slice(1, 2), slice(1, 3))
    mask[2, 1] = True
    with pytest.raises(ValueError, match="rectangular"):
        time_domain.sensor_block(mask, (4, 4))
    with pytest.raises(ValueError, match="block"):
        fdtd.gather(np.zeros((3, 10)), [[0, 0], [0, 1], [1, 1]])
    sparse = SimpleNamespace(is_sparse_grid=True)
    with pytest.raises(NotImplementedError):
        fdtd.from_fullwave(np.zeros((1, 1)), sparse, SimpleNamespace(f0=F0, c0=1500.0), density=1000.0)
