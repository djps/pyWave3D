"""The nonlinear step against the exact plane-wave solutions of the lossless Burgers equation.

With p(0, tau) = p0 sin(omega tau) and sigma = z / x_sh, x_sh = 1 / (epsilon omega p0):
- before the shock (sigma < 1), Fubini: |P_n| = 2 p0 J_n(n sigma) / (n sigma);
- well after it (sigma >> 1), the sawtooth: |P_n| -> 2 p0 / (n (1 + sigma)).
"""

import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation
from scipy.special import jv

import pywave3d as pw
from pywave3d.nonlinear import NonlinearOperator, harmonics_of, peak_pressures, waveform

F0 = 1e6
P0 = 1e6


def plane_wave(k: int, npts: int = 3) -> np.ndarray:
    f = np.zeros((k, npts, 1), dtype=complex)
    f[0] = -1j * P0  # p = Re(-i p0 e^{i w t}) = p0 sin(w t)
    return f


def march(op: NonlinearOperator, f: np.ndarray, z: float, steps: int) -> np.ndarray:
    for _ in range(steps):
        op.apply(f, z / steps)
    return f


def fubini(n: np.ndarray, sigma: float) -> np.ndarray:
    return 2 * P0 * jv(n, n * sigma) / (n * sigma)


def test_waveform_round_trip() -> None:
    rng = np.random.default_rng(0)
    h = rng.standard_normal((5, 4)) + 1j * rng.standard_normal((5, 4))
    w = waveform(h, 16)
    assert w.shape == (16, 4)
    np.testing.assert_allclose(harmonics_of(w, 5), h, atol=1e-12)
    np.testing.assert_allclose(waveform(h.T, 16, axis=-1), w.T)
    # p = Re P1 e^{i w t}: P1 = 1 gives cos
    np.testing.assert_allclose(
        waveform(np.array([[1.0 + 0j]]), 8)[:, 0], np.cos(2 * np.pi * np.arange(8) / 8), atol=1e-15
    )


@pytest.mark.parametrize("sigma", [0.3, 0.8])
def test_spectral_matches_fubini(water: pw.Medium, sigma: float) -> None:
    k = 64
    op = NonlinearOperator(water, F0, k, method="spectral")
    z = sigma * water.shock_distance(F0, P0)
    f = march(op, plane_wave(k), z, 50)
    n = np.arange(1, 6)
    np.testing.assert_allclose(np.abs(f[:5, 0, 0]), fubini(n, sigma), rtol=2e-3, atol=1e-4 * P0)


def test_spectral_conserves_energy_before_the_shock(water: pw.Medium) -> None:
    k = 64
    op = NonlinearOperator(water, F0, k, method="spectral")
    f = march(op, plane_wave(k), 0.5 * water.shock_distance(F0, P0), 20)
    assert np.sum(np.abs(f[:, 0, 0]) ** 2) == pytest.approx(P0**2, rel=1e-6)


def test_godunov_matches_fubini(water: pw.Medium) -> None:
    k = 128
    op = NonlinearOperator(water, F0, k, method="godunov")
    sigma = 0.5
    f = march(op, plane_wave(k), sigma * water.shock_distance(F0, P0), 20)
    n = np.arange(1, 4)
    np.testing.assert_allclose(np.abs(f[:3, 0, 0]), fubini(n, sigma), rtol=0.01, atol=2e-3 * P0)


def test_godunov_sawtooth_decay(water: pw.Medium) -> None:
    k = 128
    op = NonlinearOperator(water, F0, k, method="godunov")
    sigma = 4.0
    f = march(op, plane_wave(k), sigma * water.shock_distance(F0, P0), 80)
    n = np.arange(1, 4)
    np.testing.assert_allclose(np.abs(f[:3, 0, 0]), 2 * P0 / (n * (1 + sigma)), rtol=0.03)
    # the shock: the waveform jumps by about 2 pi p0 / (1 + sigma) within a few samples
    w = waveform(f[:, :1, 0], 4 * k)[:, 0]
    assert np.max(np.abs(np.roll(w, -4) - w)) > 0.8 * (2 * np.pi * P0 / (1 + sigma))


def test_auto_switches_to_godunov(water: pw.Medium) -> None:
    k = 16
    op = NonlinearOperator(water, F0, k, method="auto", switch_threshold=1e-3)
    f = plane_wave(k)
    used = set()
    xs = water.shock_distance(F0, P0)
    for _ in range(40):
        used.add(op.apply(f, xs / 20))
    assert used == {"spectral", "godunov"}
    assert op.shocked


def test_second_harmonic_grows_linearly(water: pw.Medium) -> None:
    k = 8
    op = NonlinearOperator(water, F0, k, method="spectral")
    sigma = 0.02
    f = march(op, plane_wave(k), sigma * water.shock_distance(F0, P0), 2)
    assert abs(f[1, 0, 0]) == pytest.approx(P0 * sigma / 2, rel=1e-3)


def test_zero_and_linear_cases(water: pw.Medium) -> None:
    op = NonlinearOperator(water, F0, 4)
    f = np.zeros((4, 2, 2), dtype=complex)
    assert op.apply(f, 1e-3) == "none"
    lin = NonlinearOperator(pw.Medium(nonlinearity=0), F0, 4)
    assert lin.apply(plane_wave(4), 1e-3) == "none"
    with pytest.raises((ValueError, BeartypeCallHintParamViolation)):
        NonlinearOperator(water, F0, 4, method="euler")  # type: ignore[arg-type]


@pytest.mark.parametrize("method", ["spectral", "godunov"])
def test_threads_and_chunks_do_not_change_results(water: pw.Medium, method: str) -> None:
    rng = np.random.default_rng(3)
    f = (rng.standard_normal((8, 40, 30)) + 1j * rng.standard_normal((8, 40, 30))) * 1e5
    f[0] *= 10
    a, b = f.copy(), f.copy()
    NonlinearOperator(water, F0, 8, method=method, amplitude_threshold=0.0).apply(a, 1e-3)
    op = NonlinearOperator(water, F0, 8, method=method, amplitude_threshold=0.0, workers=4, chunk_size=100)
    op.apply(b, 1e-3)
    op.close()
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-6)


def test_small_amplitudes_are_skipped(water: pw.Medium) -> None:
    f = plane_wave(8, npts=2)
    f[0, 1] *= 1e-6
    before = f[:, 1].copy()
    NonlinearOperator(water, F0, 8, amplitude_threshold=1e-4).apply(f, 1e-2)
    np.testing.assert_array_equal(f[:, 1], before)
    assert abs(f[1, 0, 0]) > 0


def rise_time(harmonics: np.ndarray, period: float, nt: int = 4096) -> tuple[float, float]:
    """(10-90 % rise time [s], jump [Pa]) of the steepest front of a waveform."""
    from pywave3d.nonlinear import waveform

    u = waveform(harmonics.reshape(-1, 1), nt)[:, 0]
    i = int(np.argmax(np.roll(u, -1) - u))
    u = np.roll(u, nt // 2 - i)
    c, w = nt // 2, nt // 4
    lo, hi = u[c - w : c].min(), u[c : c + w].max()
    jump = hi - lo
    # the crossings of the 10 % and 90 % levels nearest the front
    below = np.flatnonzero(u[:c] < lo + 0.1 * jump)
    above = np.flatnonzero(u[c:] > lo + 0.9 * jump)
    return (c + above[0] - below[-1]) * period / nt, jump


def test_adaptive_smoothing_sets_the_taylor_diffusivity() -> None:
    """A plane wave past its shock: the smoothing widens the shock to about tau_sh, with the
    diffusivity of a Taylor shock of that rise time, and acts like that diffusivity would."""
    water = pw.Medium.water()
    f0, k, p0 = 1e6, 32, 1e6
    xsh = water.shock_distance(f0, p0)
    steps = 150
    dz = 3 * xsh / steps
    target = 7 / (2 * k * f0)

    def run(medium: pw.Medium, smoothing: pw.AdaptiveSmoothing | None) -> tuple[np.ndarray, float]:
        op = pw.NonlinearOperator(medium, f0, k, method="godunov", smoothing=smoothing)
        alpha = medium.alpha(f0, np.arange(1.0, k + 1))[:, None, None]
        p = np.zeros((k, 1, 1), dtype=complex)
        p[0] = -1j * p0  # p = p0 sin(omega tau)
        for _ in range(steps):
            op.apply(p, dz)
            p *= np.exp(-alpha * dz)
        return p[:, 0, 0], op.last_artificial_diffusivity

    plain, _ = run(water, None)
    smoothed, delta_a = run(water, pw.AdaptiveSmoothing(rise_points=7))
    rise, jump = rise_time(smoothed, 1 / f0)
    assert rise_time(plain, 1 / f0)[0] < 0.3 * target  # Godunov alone keeps the shock sharp
    assert 0.5 * target < rise < 1.5 * target
    # the diffusivity is that of a Taylor shock of this jump and rise time
    taylor = target * water.nonlinearity * jump / (4 * np.arctanh(0.8) * water.density)
    assert delta_a + water.diffusivity == pytest.approx(taylor, rel=0.1)
    # and the shock is as wide as in a medium with that diffusivity
    viscous, _ = run(pw.Medium(diffusivity=delta_a + water.diffusivity), None)
    assert rise == pytest.approx(rise_time(viscous, 1 / f0)[0], rel=0.2)


def test_adaptive_smoothing_leaves_smooth_waveforms_alone() -> None:
    water = pw.Medium.water()
    op = pw.NonlinearOperator(water, 1e6, 32, method="godunov", smoothing=pw.AdaptiveSmoothing())
    p = np.zeros((32, 2, 2), dtype=complex)
    p[0] = 1e5
    q = p.copy()
    op.apply(q, 1e-3)
    assert op.last_artificial_diffusivity == 0.0
    plain = pw.NonlinearOperator(water, 1e6, 32, method="godunov")
    plain.apply(p, 1e-3)
    np.testing.assert_array_equal(p, q)
    with pytest.raises(ValueError):
        pw.AdaptiveSmoothing(rise_points=0)


def _dense_peaks(h: np.ndarray, nt: int = 1 << 14) -> tuple[np.ndarray, np.ndarray]:
    w = waveform(h, nt)
    return w.max(axis=0), w.min(axis=0)


def test_peak_pressures_find_the_peak_between_samples() -> None:
    """One harmonic peaking half-way between the 4 samples of a period: the samples alone give
    cos(45 deg) = 0.71 of the amplitude; the refined peak is the amplitude."""
    h = np.array([[np.exp(1j * np.pi / 4)]])
    assert waveform(h, 4).max() == pytest.approx(np.cos(np.pi / 4))
    pp, pm = peak_pressures(h)
    assert pp[0] == pytest.approx(1.0, abs=1e-12)
    assert pm[0] == pytest.approx(-1.0, abs=1e-12)


@pytest.mark.parametrize("k", [1, 2, 3, 8, 32])
def test_peak_pressures_match_dense_sampling(k: int) -> None:
    """Random smooth spectra (falling as 1/n) of any shape, in both precisions. Dense samples
    miss the peak too, by at most (dphi^2 / 2) sum n^2 |P_n|: the refined peak lies between the
    dense sampled one and that much above it."""
    rng = np.random.default_rng(k)
    n = np.arange(1, k + 1)[:, None, None]
    h = (rng.normal(size=(k, 5, 7)) + 1j * rng.normal(size=(k, 5, 7))) / n
    nt = 1 << 14
    ref_pp, ref_pm = _dense_peaks(h.reshape(k, -1), nt)
    miss = 0.5 * (2 * np.pi / nt) ** 2 * (n**2 * np.abs(h)).sum(axis=0).ravel()
    scale = np.abs(h).sum(axis=0).max()
    pp, pm = peak_pressures(h)
    assert pp.shape == pm.shape == (5, 7)
    eps = 1e-12 * scale
    assert np.all(pp.ravel() >= ref_pp - eps) and np.all(pp.ravel() <= ref_pp + miss + eps)
    assert np.all(pm.ravel() <= ref_pm + eps) and np.all(pm.ravel() >= ref_pm - miss - eps)
    pp32, pm32 = peak_pressures(h.astype(np.complex64), chunk=4)  # several chunks
    assert pp32.dtype == pm32.dtype == np.float64
    np.testing.assert_allclose(pp32, pp, rtol=0, atol=1e-6 * scale)
    np.testing.assert_allclose(pm32, pm, rtol=0, atol=1e-6 * scale)


def test_peak_pressures_of_a_sawtooth_are_not_below_the_samples() -> None:
    """A shocked (sawtooth) spectrum: the refined peaks are at least the sampled ones and within
    the Gibbs overshoot of the dense ones."""
    k = 64
    n = np.arange(1, k + 1)
    h = (1j / n)[:, None]  # sawtooth, sum sin(n phi) / n
    sampled = waveform(h, 4 * k)
    pp, pm = peak_pressures(h)
    ref_pp, ref_pm = _dense_peaks(h)
    assert pp[0] >= sampled.max() and pm[0] <= sampled.min()
    assert pp[0] == pytest.approx(ref_pp[0], rel=1e-3)
    assert pm[0] == pytest.approx(ref_pm[0], rel=1e-3)


def test_peak_pressures_of_nothing() -> None:
    pp, pm = peak_pressures(np.zeros((3, 0), dtype=complex))
    assert pp.shape == pm.shape == (0,)
    pp, pm = peak_pressures(np.zeros((3, 2), dtype=complex))
    np.testing.assert_array_equal(pp, 0.0)
    np.testing.assert_array_equal(pm, 0.0)


def test_peak_pressures_with_a_floor_refine_only_what_can_exceed_it() -> None:
    """With a floor, every point whose peak may exceed it is exact; the others keep their sampled
    value, which is not above the floor and within the bound of the true peak."""
    rng = np.random.default_rng(7)
    k = 6
    n = np.arange(1, k + 1)[:, None]
    h = (rng.normal(size=(k, 400)) + 1j * rng.normal(size=(k, 400))) / n**2
    exact_pp, exact_pm = peak_pressures(h)
    f_plus = np.quantile(exact_pp, 0.7) * np.ones(400)
    f_minus = float(np.quantile(-exact_pm, 0.9))
    pp, pm = peak_pressures(h, floor=(f_plus, f_minus), chunk=64)
    miss = 0.5 * (2 * np.pi / (4 * k)) ** 2 / 4 * (n**2 * np.abs(h)).sum(axis=0)
    above = exact_pp > f_plus
    np.testing.assert_allclose(pp[above], exact_pp[above], rtol=1e-13)
    assert np.all(pp[~above] <= f_plus[~above] + 1e-12) and np.all(exact_pp - pp <= miss + 1e-12)
    np.testing.assert_allclose(pm[-exact_pm > f_minus], exact_pm[-exact_pm > f_minus], rtol=1e-13)
    assert pp.max() == pytest.approx(exact_pp.max(), rel=1e-13)
    assert pm.min() == pytest.approx(exact_pm.min(), rel=1e-13)
    assert np.sum(np.abs(pp - exact_pp) > 1e-9) > 100  # most points below the floor were not refined
