"""End-to-end runs: linear focusing against Rayleigh-Sommerfeld, the focus and the outputs."""

import json

import numpy as np
import pytest

import pywave3d as pw
from conftest import focused_model, rayleigh_axis
from pywave3d.cli import main
from pywave3d.options import Options


def linear_settings(**kw) -> pw.SolverSettings:
    return pw.SolverSettings(nonlinearity=False, absorption=False, **kw)


@pytest.fixture(scope="module")
def linear_run() -> tuple[pw.WaveModel, pw.Results]:
    model = focused_model()
    grid = model.make_grid(Options({"NumberOfPointsPerWaveLength": 4, "SamplingFrequencyMultiplier": 2}))
    return model, pw.WaveSolver(model, grid, linear_settings()).run()


def test_linear_axis_matches_rayleigh_sommerfeld(linear_run) -> None:
    model, res = linear_run
    z = res.z[::10]
    ref = rayleigh_axis(model, res.grid, z)
    got = np.abs(res.axis[::10, 0])
    beyond = z > 0.25 * model.transducer.focusing_depth
    assert np.max(np.abs(got - ref)[beyond]) < 0.02 * ref[beyond].max()


def test_focus_is_identified_and_its_plane_kept(linear_run) -> None:
    _, res = linear_run
    i = res.focal_index
    assert i == int(np.argmax(res.axis_p_plus))
    assert 0 < i < res.z.size - 1
    assert res.focal_plane is not None
    ix, iy = res.grid.centre_index
    np.testing.assert_allclose(res.focal_plane[:, ix, iy], res.axis[i])
    m = res.metrics()
    assert m["focus"]["z"] == res.z[i]
    assert m["focus"]["x"] == res.grid.x[ix]
    assert not m["focus"]["on_edge"]
    assert m["spatial_peak_p_positive"]["value"] >= m["focus"]["p_positive"]
    # focal region of a piston of gain G = k a^2 / 2d: about 7 lambda (d/2a)^2 long
    assert 0 < m["focal_width_x_6dB"] == pytest.approx(m["focal_width_y_6dB"])
    assert m["focal_area_6dB"] == pytest.approx(np.pi / 4 * m["focal_width_x_6dB"] ** 2, rel=0.15)
    assert np.isfinite(m["focal_length_6dB"])


def test_lossless_linear_power_is_conserved(linear_run) -> None:
    _, res = linear_run
    # the hard-edged source holds evanescent components, which carry no power and decay in
    # the first steps; from then on the propagating power is conserved
    assert res.power[1] == pytest.approx(res.power[0], rel=0.02)
    np.testing.assert_allclose(res.power[5:], res.power[5], rtol=1e-4)


def test_merged_linear_steps_give_the_same_field() -> None:
    model = focused_model(radius=3e-3, depth=15e-3, lx=12e-3, lz=12e-3, amplitude=1e6)
    grid = model.make_grid(Options({"NumberOfPointsPerWaveLength": 4, "SamplingFrequencyMultiplier": 8}))
    every = pw.WaveSolver(model, grid, pw.SolverSettings(record_every=1)).run()
    sparse = pw.WaveSolver(model, grid, pw.SolverSettings(record_every=7, plane_peaks=False)).run()
    np.testing.assert_allclose(sparse.final, every.final, rtol=1e-9, atol=1e-9 * np.abs(every.final).max())
    assert sparse.z.size < every.z.size
    assert sparse.z[-1] == every.z[-1]


def test_nonlinear_trends_with_amplitude() -> None:
    """As the source amplitude rises (cf. tests/validation/nonlinear_trends.py): the focal
    waveform becomes asymmetric (p+ / |p-| grows from 1), the second harmonic grows in
    proportion to the amplitude, and absorption at the harmonics removes power."""
    ratios, p2 = [], []
    for amplitude in (0.05e6, 1.5e6):
        model = focused_model(radius=4e-3, depth=20e-3, lx=16e-3, lz=25e-3, amplitude=amplitude)
        grid = model.make_grid(Options({"NumberOfPointsPerWaveLength": 5, "SamplingFrequencyMultiplier": 32}))
        res = pw.WaveSolver(model, grid).run()
        m = res.metrics()
        ratios.append(m["focus"]["p_positive"] / -m["focus"]["p_negative"])
        p2.append(m["focus"]["harmonics"][1] / m["focus"]["harmonics"][0])
    assert ratios[0] == pytest.approx(1.0, abs=0.02)
    assert ratios[1] > 1.1
    assert p2[1] / p2[0] == pytest.approx(30.0, rel=0.3)  # proportional while small
    assert res.power[-1] < res.power[5]


def test_single_precision_agrees() -> None:
    model = focused_model(radius=3e-3, depth=15e-3, lx=12e-3, lz=12e-3, amplitude=1e6)
    grid = model.make_grid(Options({"NumberOfPointsPerWaveLength": 4, "SamplingFrequencyMultiplier": 8}))
    d = pw.WaveSolver(model, grid).run()
    s = pw.WaveSolver(model, grid, pw.SolverSettings(single_precision=True)).run()
    np.testing.assert_allclose(s.axis_p_plus, d.axis_p_plus, rtol=1e-3)


def test_settings_from_options() -> None:
    s = pw.SolverSettings.from_options(
        Options({"Options": {"NLN": False, "Adsorp": False}, "DiffractSolver": {"Type": "parabolic"}})
    )
    assert not s.nonlinearity and not s.absorption and s.paraxial and s.diffraction
    d = pw.SolverSettings.from_options(Options({}))
    assert d.nonlinearity and d.absorption and d.diffraction and not d.absorbing_layer
    assert d.progress_bar
    assert not pw.SolverSettings.from_options(Options({"Output": {"progress_bar": False}})).progress_bar
    with pytest.raises(TypeError):
        pw.SolverSettings.from_options(Options({"DiffractSolver": {"Type": "2DSym"}}))


def test_reference_options_run(repo, tmp_path) -> None:
    res = pw.simulate(repo / "options" / "woptions.json")
    files = res.save(tmp_path)
    names = {f.name for f in files}
    assert {"axial.txt", "metrics.json", "results.npz", "focal_plane.txt", "p0.txt"} <= names
    assert "p4.txt" not in names  # z = 1 m is beyond the 20 mm domain
    p0 = np.loadtxt(tmp_path / "p0.txt")
    assert p0.shape[1] == 3  # t, p, instantaneous intensity
    axial = np.loadtxt(tmp_path / "axial.txt")
    np.testing.assert_allclose(axial[:, 0], res.z)


def test_cli(tmp_path, capsys) -> None:
    opts = {
        "Model": {
            "Geometry": {"lx": 0.01, "ly": 0.01, "lz": 0.01},
            "Transducer": {"outer_radius": 0.003, "focusing_depth": 0.012, "power": 5},
        },
        "Solver": {
            "DiffractSolver": {"NumberOfPointsPerWaveLength": 3, "SamplingFrequencyMultiplier": 8},
            "Output": {"xyz": {"name": "pf.txt", "position": [0.005, 0.005, 0.01]}},
        },
    }
    f = tmp_path / "o.json"
    f.write_text(json.dumps(opts))
    assert main(["2", str(f), "-o", str(tmp_path / "out"), "-q"]) == 0
    assert capsys.readouterr().out == ""
    met = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert met["focus"]["p_positive"] > 0
    assert (tmp_path / "out" / "pf.txt").exists()
    assert main(["1", str(tmp_path / "missing.json")]) == 1


def test_beartype_checks_arguments() -> None:
    from beartype.roar import BeartypeCallHintParamViolation

    with pytest.raises(BeartypeCallHintParamViolation):
        pw.Grid.from_lengths("1", 1.0, 1.0, wavelength=1e-3)  # type: ignore[arg-type]
    with pytest.raises(BeartypeCallHintParamViolation):
        pw.LinearPropagator(pw.Grid(4, 4, 4, 2, 1e-3, 1e-3, 1e-3), pw.Medium(), 1e6).apply(
            np.ones((1, 4, 4))
        )  # real field


def test_pruned_plane_peaks_equal_brute_force() -> None:
    """Plane extremes are computed only where sum |P_n| can beat the known values."""
    from pywave3d.nonlinear import waveform

    model = focused_model(radius=3e-3, depth=15e-3, lx=12e-3, lz=12e-3, amplitude=1e6)
    grid = model.make_grid(Options({"NumberOfPointsPerWaveLength": 4, "SamplingFrequencyMultiplier": 8}))
    res = pw.WaveSolver(model, grid).run()
    w = waveform(res.final, 4 * res.final.shape[0])
    assert res.plane_p_plus[-1] == pytest.approx(w.max(axis=0).max(), rel=1e-9)
    assert res.plane_p_minus[-1] == pytest.approx(w.min(), rel=1e-9)
    assert res.max_p_plus is not None
    assert np.all(res.max_p_plus >= w.max(axis=0) - 1e-6 * w.max())
    assert res.max_p_plus.max() == pytest.approx(np.nanmax(res.plane_p_plus))


def test_cli_progress_flag(monkeypatch) -> None:
    import pywave3d.cli as cli

    seen = []

    def fake_simulate(opts, workers, progress_bar):
        seen.append(progress_bar)
        raise ValueError("stop here")  # main reports it and returns 1

    monkeypatch.setattr(cli, "simulate", fake_simulate)
    assert cli.main(["1"]) == 1 and seen.pop() is None  # None: the options decide
    assert cli.main(["1", "--no-progress"]) == 1 and seen.pop() is False


def test_shock_smoothing_settings_and_run() -> None:
    assert pw.SolverSettings.from_options(Options({})).shock_smoothing is None
    assert pw.SolverSettings.from_options(Options({"ShockSmoothing": {"enabled": False}})).shock_smoothing is None
    s = pw.SolverSettings.from_options(
        Options({"ShockSmoothing": {"peak_multiplier": 20, "post_focal_multiplier": 2, "start_width": 0.02}})
    )
    sm = s.shock_smoothing
    assert sm is not None
    assert (sm.peak_multiplier, sm.post_focal_multiplier, sm.start, sm.end) == (20, 2, 0.96, 1.03)
    assert (sm.start_width, sm.end_width, sm.focal_length) == (0.02, 0.01, 0.0)
    adaptive = pw.SolverSettings.from_options(Options({"ShockSmoothing": {"type": "adaptive", "rise_points": 9}}))
    assert adaptive.shock_smoothing == pw.AdaptiveSmoothing(rise_points=9)
    with pytest.raises(TypeError):
        pw.SolverSettings.from_options(Options({"ShockSmoothing": {"type": "spatial"}}))

    # a linear run: the smoothing lowers the field only from the start of the profile on
    model = focused_model()
    grid = model.make_grid(Options({"NumberOfPointsPerWaveLength": 4, "SamplingFrequencyMultiplier": 4}))
    plain = pw.WaveSolver(model, grid, pw.SolverSettings(nonlinearity=False)).run()
    smoothed = pw.WaveSolver(
        model,
        grid,
        pw.SolverSettings(nonlinearity=False, shock_smoothing=pw.AxialSmoothing(peak_multiplier=1e4)),
    ).run()
    zf = model.transducer.focusing_depth
    before = plain.z - grid.z0 < 0.9 * zf
    after = plain.z - grid.z0 > 1.05 * zf
    a0, a1 = np.abs(plain.axis[:, 0]), np.abs(smoothed.axis[:, 0])
    np.testing.assert_allclose(a1[before], a0[before], rtol=1e-6)
    assert np.all(a1[after] < 0.99 * a0[after])
