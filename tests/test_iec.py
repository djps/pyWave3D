"""The IEC field parameters (pywave3d.iec) against analytic cases, and in a full run."""

import math

import numpy as np
import pytest

import pywave3d as pw
from conftest import focused_model
from pywave3d import iec
from pywave3d.nonlinear import waveform
from pywave3d.options import Options

F0 = 1e6
Z0 = 1.479e6


def test_derating() -> None:
    # 0.3 dB/(cm MHz) at 2 MHz over 5 cm: 3 dB in pressure
    assert iec.derating(2e6, np.array([0.05]))[0] == pytest.approx(10 ** (-3 / 20))
    assert iec.derating(1e6, np.array([0.0]))[0] == 1.0


def test_working_frequencies() -> None:
    sine = np.array([1.0 + 0j, 0, 0, 0])
    assert iec.working_frequencies(sine, F0) == {"arithmetic_mean": F0, "zero_crossing": F0}
    second = np.array([0.1 + 0j, 1.0, 0, 0])
    assert iec.working_frequencies(second, F0)["arithmetic_mean"] == 2 * F0
    assert iec.working_frequencies(second, F0)["zero_crossing"] == 2 * F0


def test_pulse_integrals_by_parseval() -> None:
    h = np.array([2.0 + 1j, 0.5 - 0.3j, 0.1j])
    ppsi, pii = iec.pulse_integrals(h, F0, Z0)
    w = waveform(h.reshape(-1, 1), 64)[:, 0]
    assert float(ppsi) == pytest.approx(np.mean(w**2) / F0, rel=1e-12)
    assert float(pii) == pytest.approx(float(ppsi) / Z0)


def test_active_intensity(small_grid: pw.Grid) -> None:
    c, rho = 1482.0, 998.0
    k = 2 * math.pi * F0 / c
    h = np.zeros((small_grid.n_harmonics, small_grid.nx, small_grid.ny), dtype=complex)
    h[0] = 1e5
    plane = iec.active_intensity(h, small_grid, F0, c, rho)
    np.testing.assert_allclose(plane, 1e10 / (2 * rho * c))
    # an oblique plane wave carries cos(theta) of the plane-wave estimate
    m = 3
    h[0] = 1e5 * np.exp(1j * small_grid.kx[m] * small_grid.x)[:, None]
    cos = math.sqrt(1 - (small_grid.kx[m] / k) ** 2)
    np.testing.assert_allclose(iec.active_intensity(h, small_grid, F0, c, rho), cos * 1e10 / (2 * rho * c))
    # an evanescent one carries none
    h[0] = np.exp(1j * small_grid.kx[12] * small_grid.x)[:, None]
    np.testing.assert_allclose(iec.active_intensity(h, small_grid, F0, c, rho), 0.0, atol=1e-20)


def test_area_holding() -> None:
    v = np.zeros((10, 10))
    v[:2, :5] = 1.0
    assert iec.area_holding(v, 2.0, 0.75) == pytest.approx(2.0 * 8)
    assert iec.area_holding(np.zeros((3, 3)), 1.0, 0.5) == 0.0


def test_spatial_average(small_grid: pw.Grid) -> None:
    h = np.full((2, small_grid.nx, small_grid.ny), 3.0 + 1j)
    np.testing.assert_allclose(iec.spatial_average(h, small_grid, 1e-3), h)
    xc, yc = small_grid.centre
    r2 = (small_grid.x[:, None] - xc) ** 2 + (small_grid.y[None, :] - yc) ** 2
    g = np.exp(-r2 / (2 * (1e-3) ** 2))[None].astype(complex)
    avg = iec.spatial_average(g, small_grid, 1.5e-3)
    disc = r2 <= (0.75e-3) ** 2
    ix, iy = small_grid.centre_index
    assert avg[0, ix, iy].real == pytest.approx(g[0][disc].real.mean())
    assert iec.spatial_average(g, small_grid, 0.0) is g


def test_secondary_maxima_of_sinc() -> None:
    x = np.linspace(-6, 6, 2401)
    v = np.abs(np.sinc(x))  # a pressure-like profile: levels are 20 log10
    s = iec.secondary_maxima(x, v, 1200)
    assert s["after_level_dB"] == pytest.approx(-13.26, abs=0.02)
    assert s["before_position"] == pytest.approx(-1.4303, abs=0.01)
    mono = iec.secondary_maxima(x, np.exp(-(x**2)), 1200)
    assert math.isnan(mono["after_value"])


def test_beam_alignment_axis() -> None:
    z = np.linspace(0, 0.05, 30)
    a = iec.beam_alignment_axis(z, 1e-3 + 0.02 * z, 2e-3 - 0.01 * z)
    assert a["x0"] == pytest.approx(1e-3) and a["slope_x"] == pytest.approx(0.02)
    assert a["y0"] == pytest.approx(2e-3) and a["slope_y"] == pytest.approx(-0.01)
    assert a["tilt"] == pytest.approx(math.atan(math.hypot(0.02, 0.01)))


def test_shock_parameters() -> None:
    k = 64
    sine = np.zeros(k, dtype=complex)
    sine[0] = -1j
    s = iec.shock_parameters(sine, F0)
    assert s["steepness"] == pytest.approx(1.0, rel=1e-3)
    assert not s["shocked"]
    n = np.arange(1, k + 1)
    saw = 2j / (math.pi * n) * (-1.0) ** (n + 1)  # unit sawtooth, rising jump of 2
    s = iec.shock_parameters(saw, F0)
    assert s["shocked"]
    assert s["shock_amplitude"] == pytest.approx(2.0, rel=0.2)


def test_mechanical_and_thermal_indices() -> None:
    z = np.linspace(0, 0.1, 10001)
    p_r = 2e6 * np.exp(-(((z - 0.04) / 0.01) ** 2))
    pii = p_r**2 / Z0 / F0
    mi = iec.mechanical_index(z, p_r, pii, 2e6)
    zi = mi["z_MI"]
    assert zi < 0.04  # derating moves the maximum of pii towards the source
    assert mi["MI"] == pytest.approx(mi["p_r_derated"] / 1e6 / math.sqrt(2))
    ti = iec.thermal_indices(z, 0.1, np.full(z.size, 1e4), 2e6, 0.5e-4)  # 100 mW, 0.5 cm^2
    assert ti["TIS"] == pytest.approx(100 * 2 / iec.C_TIS)
    d_eq = math.sqrt(4 * 0.5 / math.pi)
    assert ti["TIC"] == pytest.approx(100 / (d_eq * iec.C_TIC))
    assert ti["z_bp"] == pytest.approx(1.5 * d_eq / 100)
    assert ti["TIB"] > 0


@pytest.fixture(scope="module")
def run() -> pw.Results:
    model = focused_model(radius=4e-3, depth=20e-3, lx=16e-3, lz=25e-3, amplitude=1e6)
    grid = model.make_grid(Options({"NumberOfPointsPerWaveLength": 5, "SamplingFrequencyMultiplier": 16}))
    return pw.WaveSolver(model, grid, pw.SolverSettings(hydrophone_diameter=3e-3)).run()


def test_every_iec_output_is_recorded(run: pw.Results) -> None:
    m = run.metrics()
    e = m["iec"]
    for key in (
        "p_c",
        "p_r",
        "I_ta_focus",
        "I_ta_focus_exact",
        "I_spta",
        "I_sppa",
        "pii_focus",
        "ppsi_focus",
        "power_in_focal_area_6dB",
        "power_in_focal_area_6dB_exact",
    ):
        assert math.isfinite(e[key]) and e[key] > 0, key
    assert e["acoustic_working_frequency"]["arithmetic_mean"] == F0
    for lv in ("3", "6", "12", "20"):
        assert f"focal_width_x_{lv}dB" in m and f"focal_area_{lv}dB" in m
    assert m["focal_area_3dB"] < m["focal_area_6dB"] < m["focal_area_12dB"]
    # the transducer is on the axis: the fitted beam axis is not tilted
    assert e["beam_alignment_axis"]["tilt"] < 0.01
    assert e["beam_alignment_axis"]["x0"] == pytest.approx(run.grid.centre[0], abs=run.grid.dx)
    # exact power below the plane-wave estimate by the obliquity, and conserved but for absorption
    ex = e["power"]["exact"]
    assert 0.9 * e["power"]["focal_plane"] < ex["focal_plane"] <= e["power"]["focal_plane"]
    assert ex["end"] < ex["focal_plane"] < ex["source"] * 1.001
    assert 0 < e["power_in_focal_area_6dB"] < e["power"]["focal_plane"]
    # a hydrophone averages the focus down
    assert e["hydrophone"]["p_c"] < e["p_c"]
    d = e["derated"]
    assert d["MI"] < m["focus"]["mechanical_index_water"] * 1.0001
    assert all(math.isfinite(d[k]) for k in ("TIS", "TIB", "TIC", "A_aprt"))
    assert d["A_aprt"] == pytest.approx(math.pi * 4e-3**2, rel=0.15)
    ph = e["physiotherapy"]
    assert ph["z"] == pytest.approx(3e-3, abs=run.grid.dz)
    assert 0 < ph["A_BCS"] < ph["A_ER"] and ph["R_BN"] > 1
    assert e["nonlinear"]["p_c_over_p_r"] > 1 and e["nonlinear"]["P2_over_P1"] > 0
    assert "side_lobes" in e and "axial" in e["side_lobes"]


def test_files(run: pw.Results, tmp_path) -> None:
    names = {f.name for f in run.save(tmp_path)}
    assert {"axial.txt", "planes.txt", "focal_plane.txt", "metrics.json"} <= names
    ax = np.loadtxt(tmp_path / "axial.txt")
    assert ax.shape[1] == 8 + 4
    np.testing.assert_allclose(ax[:, 5], run.axial_pii)
    pl = np.loadtxt(tmp_path / "planes.txt")
    np.testing.assert_allclose(pl[:, 1], run.power)


def test_ripples_on_the_main_lobe_are_not_side_lobes() -> None:
    x = np.linspace(-6, 6, 2401)
    v = np.abs(np.sinc(x)) * (1 + 0.01 * np.cos(40 * x))
    s = iec.secondary_maxima(x, v, 1200)
    assert s["after_level_dB"] == pytest.approx(-13.26, abs=0.15)
    assert s["after_position"] > 1.0


def test_every_key_in_the_document_is_produced(run: pw.Results) -> None:
    """docs/iec_outputs.md names the metrics.json keys of each parameter (`iec.<key>`, apart
    from the functions of pywave3d.iec it also names): they must exist."""
    import re
    from pathlib import Path

    doc = (Path(__file__).parents[1] / "docs" / "iec_outputs.md").read_text(encoding="utf-8")
    m = run.metrics()
    names = {n for n in re.findall(r"`(iec\.[A-Za-z0-9_.]+)`", doc) if not hasattr(iec, n.split(".")[1])}
    assert len(names) > 20
    for name in sorted(names):
        node = m
        for part in name.split("."):
            if part == "...":
                break
            assert isinstance(node, dict) and part in node, f"{name}: no '{part}'"
            node = node[part]
