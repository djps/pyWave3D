"""Linear propagation against the continuous-wave angular spectrum solver of k-Wave.

``kwave.utils.angular_spectrum_cw`` (the Python port of k-Wave's ``angularSpectrumCW``) projects
a CW source plane to given z positions in one step per plane, with zero-padding and the angular
restriction of Zeng and McGough (2008). pyWave3D marches step by step in retarded time, so
P_pywave3d(z) = P_kwave(z) exp(+i k z).

Three comparisons on a focused piston in water:

1. the same periodic problem (no padding, no angular restriction): the two must agree to
   round-off, which checks the propagator, its sign convention and the evanescent branch;
2. k-Wave's defaults (zero-padded, angularly restricted) on a wide domain: agreement to 2% of
   the peak on the axis and over the focal plane (the difference is the periodic images of
   pyWave3D's domain), and pyWave3D at least as close as k-Wave to the Rayleigh-Sommerfeld
   integral (both are about 1.3% from it, from the band-limited hard-edged source);
3. with absorption (alpha = alpha0 (f / f0)^2): k-Wave applies exp(-alpha z k / kz), pyWave3D
   exp(-alpha z); they agree to 1.5% of the peak.

Skipped if k-wave-python (``pip install k-wave-python``) is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

import pywave3d as pw
from conftest import rayleigh_axis
from pywave3d.options import Options

kw = pytest.importorskip("kwave.utils.angular_spectrum_cw")

F0 = 1_000_000
C0 = 1482.0


def angular_spectrum_cw(plane: np.ndarray, dx: float, z: np.ndarray, medium: float | dict, **kwargs) -> np.ndarray:
    """k-Wave's CW solver; its type hints ask for a real plane, so the unchecked function is used for a complex one."""
    fn = getattr(kw.angular_spectrum_cw, "__wrapped__", kw.angular_spectrum_cw)
    return fn(plane, dx, z, F0, medium, loops_for_time_est=10**9, **kwargs)


def run(lx: float, alpha0: float = 0.0, nz: int | None = None) -> tuple[pw.WaveModel, pw.Results]:
    medium = pw.Medium(sound_speed=C0, attenuation=pw.Attenuation("power_law", alpha0=alpha0, power=2.0))
    model = pw.WaveModel(
        medium=medium,
        frequency=float(F0),
        lengths=(lx, lx, 40e-3),
        transducer=pw.Transducer(outer_radius=6e-3, focusing_depth=30e-3, amplitude=1e5),
    )
    grid = model.make_grid(Options({"NumberOfPointsPerWaveLength": 4, "SamplingFrequencyMultiplier": 2}))
    settings = pw.SolverSettings(nonlinearity=False, absorption=alpha0 > 0, record_every=1)
    return model, pw.WaveSolver(model, grid, settings).run()


def kwave_field(
    model: pw.WaveModel, res: pw.Results, z_index: np.ndarray, medium: float | dict, **kwargs
) -> np.ndarray:
    g = res.grid
    src = model.initial_field(g)[0]
    z = res.z[z_index] - g.z0
    out = angular_spectrum_cw(src, g.dx, z, medium, **kwargs)  # (nx, ny, nz)
    k = 2 * np.pi * F0 / C0
    return np.moveaxis(out, -1, 0) * np.exp(1j * k * z)[:, None, None]  # to retarded time


def test_same_periodic_problem_agrees_to_round_off() -> None:
    model, res = run(lx=20e-3)
    g = res.grid
    zi = np.array([10, 60, res.z.size - 1])
    ref = kwave_field(model, res, zi, C0, angular_restriction=False, fft_length=g.nx)
    ix, iy = g.centre_index
    np.testing.assert_allclose(res.axis[zi, 0], ref[:, ix, iy], rtol=1e-9, atol=1e-9 * np.abs(ref).max())
    np.testing.assert_allclose(res.final[0], ref[-1], rtol=1e-8, atol=1e-8 * np.abs(ref).max())


def test_kwave_defaults_on_a_wide_domain() -> None:
    model, res = run(lx=40e-3)
    g = res.grid
    zi = np.arange(10, res.z.size, 10)
    ref = kwave_field(model, res, zi, C0)  # zero-padded, angular restriction
    ix, iy = g.centre_index
    axis_ref = np.abs(ref[:, ix, iy])
    ours = np.abs(res.axis[zi, 0])
    # the two differ by the periodic images of pyWave3D's domain (1.3% here, 0.9% at 60 mm)
    assert np.max(np.abs(ours - axis_ref)) < 0.02 * axis_ref.max()
    # and pyWave3D is at least as close to the exact Rayleigh-Sommerfeld integral as k-Wave
    rs = rayleigh_axis(model, g, res.z[zi])
    assert np.max(np.abs(ours - rs)) <= 1.05 * np.max(np.abs(axis_ref - rs))
    # the focal plane
    i = res.focal_index
    focal = kwave_field(model, res, np.array([i]), C0)[0]
    assert res.focal_plane is not None
    assert np.max(np.abs(np.abs(res.focal_plane[0]) - np.abs(focal))) < 0.02 * np.abs(focal).max()
    # and the same focus
    zk = zi[int(np.argmax(axis_ref))]
    assert abs(res.z[zk] - res.z[i]) <= 10 * g.dz


def test_absorption_agrees() -> None:
    alpha_db = 0.5  # dB / (MHz^2 cm), much more than water so that it matters
    alpha0 = alpha_db / 8.686 * 100  # Np/m at 1 MHz
    model, res = run(lx=40e-3, alpha0=alpha0)
    g = res.grid
    zi = np.arange(10, res.z.size, 10)
    ref = kwave_field(model, res, zi, {"sound_speed": C0, "alpha_coeff": alpha_db, "alpha_power": 2.0})
    ix, iy = g.centre_index
    axis_ref = np.abs(ref[:, ix, iy])
    assert np.max(np.abs(np.abs(res.axis[zi, 0]) - axis_ref)) < 0.015 * axis_ref.max()


def test_time_domain_kwave_through_the_preprocessor() -> None:
    """A k-Wave time-domain CW run, reduced by ``preprocess.time_domain.from_kwave``.

    A 3.5 mm piston drives a 40 x 24 x 24 grid at 4 points per wavelength. The harmonics fitted
    on k-Wave's first recorded plane are marched by pyWave3D's propagator, zero-padded to three
    times the width so that it has no periodic images (k-Wave has a PML), and compared with
    k-Wave's own axis: within 6% of the peak (measured 4.1%; with pyWave3D's periodic domain
    of the same 9 mm width the images make it 35%).
    """
    kfo = pytest.importorskip("kwave.kspaceFirstOrder")
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.ksensor import kSensor
    from kwave.ksource import kSource

    from pywave3d.preprocess import time_domain

    c0, rho, f0 = 1500.0, 1000.0, 1e6
    nx, nyz = 40, 24
    dx = c0 / f0 / 4
    kgrid = kWaveGrid([nx, nyz, nyz], [dx, dx, dx])
    dt = 0.3 * dx / c0
    kgrid.setTime(int((nx * dx / c0 + 6 / f0) / dt), dt)
    source = kSource()
    mask = np.zeros((nx, nyz, nyz), bool)
    yy, zz = np.meshgrid(kgrid.y_vec.ravel(), kgrid.z_vec.ravel(), indexing="ij")
    mask[1] = np.hypot(yy, zz) <= 3.5e-3
    source.p_mask = mask
    t = np.arange(kgrid.Nt) * dt
    source.p = (1e5 * np.clip(t * f0 / 2, 0, 1) * np.sin(2 * np.pi * f0 * t))[None, :]
    sensor = kSensor(mask=np.ones((nx, nyz, nyz), bool))
    sensor.mask[:3] = False
    sensor.record = ["p"]
    data = kfo.kspaceFirstOrder(
        kgrid, kWaveMedium(sound_speed=c0, density=rho), source, sensor, pml_size=10, quiet=True
    )
    res = time_domain.from_kwave(
        data, kgrid, frequency=f0, sound_speed=c0, density=rho, sensor_mask=sensor.mask, n_harmonics=1
    )
    assert res.z.size == nx - 3 and res.grid.nx == nyz
    g = res.grid
    ix, iy = res.axis_ij
    n = 3 * g.nx
    big = np.zeros((1, n, n), complex)
    big[0, g.nx : 2 * g.nx, g.ny : 2 * g.ny] = res.source_plane[0]
    lin = pw.LinearPropagator(
        pw.Grid(nx=n, ny=n, nz=2, nt=2, dx=g.dx, dy=g.dy, dz=dx),
        pw.Medium(sound_speed=c0, density=rho),
        f0,
        absorption=False,
    )
    ours = [abs(big[0, g.nx + ix, g.ny + iy])]
    for _ in range(res.z.size - 1):
        lin.apply(big)
        ours.append(abs(big[0, g.nx + ix, g.ny + iy]))
    kwave_axis = np.abs(res.axis[:, 0])
    assert np.max(np.abs(np.array(ours) - kwave_axis)) < 0.06 * kwave_axis.max()
    m = res.metrics()
    assert m["iec"]["power"]["exact"]["source"] > 0 and m["focus"]["p_positive"] > 1e5
