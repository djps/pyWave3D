"""End-to-end runs: linear focusing against Rayleigh-Sommerfeld, the focus and the outputs."""

import collections
import json

import numpy as np
import pytest
import scipy.fft as sfft

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
    # the field: single precision rounds at ~1e-7, and the march adds little to that
    scale = np.abs(d.axis[:, 0]).max()
    np.testing.assert_allclose(s.axis, d.axis, rtol=0, atol=1e-5 * scale)
    np.testing.assert_allclose(s.final, d.final, rtol=0, atol=1e-5 * np.abs(d.final[0]).max())
    ms, md = s.metrics()["focus"], d.metrics()["focus"]
    assert ms["z"] == md["z"]
    assert ms["p_positive"] == pytest.approx(md["p_positive"], rel=1e-5)
    assert ms["p_negative"] == pytest.approx(md["p_negative"], rel=1e-5)
    # (not axis_p_plus near the source: the recorder samples the waveform on 4 x n_active
    # points, and single precision brings the harmonics in a little later than double, its
    # tolerance being 16 epsilons; that moves the sampled peak by up to ~1% where few
    # harmonics are active)


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
    from pywave3d.nonlinear import peak_pressures

    model = focused_model(radius=3e-3, depth=15e-3, lx=12e-3, lz=12e-3, amplitude=1e6)
    grid = model.make_grid(Options({"NumberOfPointsPerWaveLength": 4, "SamplingFrequencyMultiplier": 8}))
    res = pw.WaveSolver(model, grid).run()
    pp, pm = peak_pressures(res.final)
    assert res.plane_p_plus[-1] == pytest.approx(pp.max(), rel=1e-9)
    assert res.plane_p_minus[-1] == pytest.approx(pm.min(), rel=1e-9)
    assert res.max_p_plus is not None
    assert np.all(res.max_p_plus >= pp - 1e-9 * pp.max())
    assert res.max_p_plus.max() == pytest.approx(np.nanmax(res.plane_p_plus))


def test_linear_peak_pressures_are_the_amplitude() -> None:
    """In a linear run p+ = -p- = |P_1| everywhere: the recorder finds the peak of the waveform,
    not the largest of a few samples (with one harmonic, 4 samples a period miss it by up to
    29%)."""
    model = focused_model()
    grid = model.make_grid(Options({"NumberOfPointsPerWaveLength": 4, "SamplingFrequencyMultiplier": 2}))
    res = pw.WaveSolver(model, grid, linear_settings(progress_bar=False)).run()
    amp = np.abs(res.axis[:, 0])
    np.testing.assert_allclose(res.axis_p_plus, amp, rtol=1e-10)
    np.testing.assert_allclose(res.axis_p_minus, -amp, rtol=1e-10)
    m = res.metrics()
    ix, iy = res.grid.centre_index
    assert m["focus"]["p_positive"] == pytest.approx(abs(res.focal_plane[0, ix, iy]), rel=1e-10)
    maps = res.plane_maps(res.final)
    np.testing.assert_allclose(maps["p_positive"], np.abs(res.final[0]), rtol=1e-9, atol=1e-9 * amp.max())


def test_recorded_peaks_do_not_depend_on_the_active_harmonics() -> None:
    """The harmonic tolerance changes how many harmonics are active near the source, and with it
    how finely a sampled waveform would be resolved; the recorded p+ must not change with it."""
    model = focused_model(radius=3e-3, depth=15e-3, lx=12e-3, lz=12e-3, amplitude=1e6)
    grid = model.make_grid(Options({"NumberOfPointsPerWaveLength": 4, "SamplingFrequencyMultiplier": 8}))
    ref = pw.WaveSolver(model, grid, pw.SolverSettings(progress_bar=False)).run()
    late = pw.WaveSolver(model, grid, pw.SolverSettings(progress_bar=False, harmonic_tolerance=1e-6)).run()
    np.testing.assert_allclose(late.axis_p_plus, ref.axis_p_plus, rtol=1e-5)
    np.testing.assert_allclose(late.axis_p_minus, ref.axis_p_minus, rtol=1e-5)


def test_cli_progress_flag(monkeypatch) -> None:
    import pywave3d.cli as cli

    seen = []

    def fake_simulate(opts, workers, progress_bar, precision=None):
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


def _spy_ffts(monkeypatch) -> collections.Counter:
    """Count the scipy.fft calls of a run by (function, input dtype, output dtype)."""
    seen: collections.Counter = collections.Counter()
    for name in ("fft2", "ifft2", "rfft", "irfft"):
        f = getattr(sfft, name)

        def spy(x, *a, _f=f, _n=name, **k):
            out = _f(x, *a, **k)
            seen[(_n, np.asarray(x).dtype, out.dtype)] += 1
            return out

        monkeypatch.setattr(sfft, name, spy)
    return seen


@pytest.mark.parametrize("single", [True, False])
@pytest.mark.parametrize("smoothing", ["axial", "adaptive"])
def test_the_whole_march_runs_at_the_chosen_precision(monkeypatch, single, smoothing) -> None:
    """Nonlinear, with a PML and shock smoothing: every operator, every FFT and the field stay at
    the chosen precision (numpy and scipy >= 2 keep complex64 in the FFTs, so a stray
    double-precision operand would be the only way to upcast)."""
    complex_t = np.dtype(np.complex64 if single else np.complex128)
    real_t = np.dtype(np.float32 if single else np.float64)
    base = focused_model(radius=3e-3, depth=15e-3, lx=12e-3, lz=16e-3, amplitude=2e6)
    model = pw.WaveModel(
        lengths=base.lengths,
        transducer=base.transducer,
        layer=pw.AbsorbingLayer.for_reflection(1.5e-3, reflection=1e-4, kind="pml"),
    )
    grid = model.make_grid(Options({"NumberOfPointsPerWaveLength": 4, "SamplingFrequencyMultiplier": 16}))
    sm = pw.AxialSmoothing(peak_multiplier=5) if smoothing == "axial" else pw.AdaptiveSmoothing()
    settings = pw.SolverSettings(single_precision=single, absorbing_layer=True, shock_smoothing=sm,
                                 method="auto" if smoothing == "axial" else "godunov", progress_bar=False)  # fmt: skip
    solver = pw.WaveSolver(model, grid, settings)
    lin = solver.linear
    assert lin._full.dtype == lin._half.dtype == complex_t
    for pml in (lin._pml_full, lin._pml_half):
        assert pml is not None
        assert {a.dtype for a in (*pml.decay, *pml.weights)} == {real_t}
    seen = _spy_ffts(monkeypatch)
    res = solver.run()
    assert solver.field.dtype == complex_t
    assert seen
    for (_, dt_in, dt_out), _n in seen.items():
        assert {dt_in, dt_out} <= {complex_t, real_t}
    assert ("rfft", real_t, complex_t) in seen  # the nonlinear step ran
    assert np.all(np.isfinite(res.axis_p_plus))


def test_precision_from_options_cli_and_simulate(monkeypatch, tmp_path) -> None:
    assert not pw.SolverSettings.from_options(Options({})).single_precision
    assert pw.SolverSettings.from_options(Options({"precision": "Single"})).single_precision
    assert not pw.SolverSettings.from_options(Options({"precision": "double"})).single_precision
    with pytest.raises(TypeError, match="precision"):
        pw.SolverSettings.from_options(Options({"precision": "half"}))

    used: list[bool] = []
    run = pw.WaveSolver.run

    def spy(self):
        used.append(self.settings.single_precision)
        assert self.field.dtype == (np.complex64 if self.settings.single_precision else np.complex128)
        return run(self)

    monkeypatch.setattr(pw.WaveSolver, "run", spy)
    opts = {
        "Model": {"Geometry": {"lx": 0.008, "ly": 0.008, "lz": 0.004},
                  "Transducer": {"outer_radius": 0.002, "focusing_depth": 0.01, "power": 1}},
        "Solver": {"DiffractSolver": {"NumberOfPointsPerWaveLength": 3, "SamplingFrequencyMultiplier": 4},
                   "precision": "single"},
    }  # fmt: skip
    f = tmp_path / "o.json"
    f.write_text(json.dumps(opts))
    out = str(tmp_path / "out")
    assert main(["1", str(f), "-o", out, "-q", "--no-progress"]) == 0  # the file: single
    assert main(["1", str(f), "-o", out, "-q", "--no-progress", "--precision", "double"]) == 0  # the flag wins
    opts["Solver"].pop("precision")
    f.write_text(json.dumps(opts))
    assert main(["1", str(f), "-o", out, "-q", "--no-progress"]) == 0  # the default: double
    assert main(["1", str(f), "-o", out, "-q", "--no-progress", "--precision", "single"]) == 0
    assert used == [True, False, False, True]
    with pytest.raises(SystemExit):
        main(["1", str(f), "--precision", "half"])
    with pytest.raises(Exception, match="precision"):  # beartype's check, or simulate's own without it
        pw.simulate(Options(opts), precision="half")  # type: ignore[arg-type]


def test_single_precision_does_not_activate_harmonics_on_rounding_noise() -> None:
    """Single precision brings the harmonics in as the nonlinearity makes them, as double does,
    rather than all of them at once on its rounding noise."""
    model = focused_model(radius=3e-3, depth=15e-3, lx=12e-3, lz=12e-3, amplitude=1e6)
    grid = model.make_grid(Options({"NumberOfPointsPerWaveLength": 4, "SamplingFrequencyMultiplier": 64}))
    active = {}
    for single in (False, True):
        solver = pw.WaveSolver(model, grid, pw.SolverSettings(single_precision=single, progress_bar=False))
        counts = []
        count = solver._active_harmonics

        def spy(count=count, counts=counts):
            counts.append(count())
            return counts[-1]

        solver._active_harmonics = spy
        solver.run()
        active[single] = counts
    assert solver.harmonic_tolerance == pytest.approx(16 * np.finfo(np.float32).eps)
    assert max(active[False]) < grid.n_harmonics
    # single precision does not fill the top harmonics with its noise: never more active
    # harmonics than double, whose tolerance (1e-10) is lower
    assert len(active[True]) == len(active[False])
    assert np.all(np.asarray(active[True]) <= np.asarray(active[False]))
