"""The nonlinear part of a step: the plane-wave (Burgers) equation in retarded time.

At each transverse point the waveform obeys, over one step,

    dp/dz = epsilon p dp/dtau,      epsilon = beta / (rho0 c0^3),

(the nonlinear term of the Westervelt and KZK equations), so positive pressures advance to
earlier retarded times and the waveform steepens into shocks. Two integrators are provided,
as in the MSU HIFU codes (Yuldashev and Khokhlova 2011):

``"spectral"``
    in the frequency domain, on the harmonics. With p = Re sum P_n exp(+i n omega tau),

        dP_n/dz = (epsilon / 2) (i n omega) [p^2]_n,

    where [p^2]_n is evaluated pseudo-spectrally (inverse FFT, square, FFT) on a grid of
    M > 3K samples, so that the quadratic product is free of aliasing, and integrated with
    the classical 4th-order Runge-Kutta scheme. Accurate and cheap while the spectrum decays
    well before the K-th harmonic; once shocks form, energy piles up at the top harmonics.

``"godunov"``
    in the time domain, on the waveform: the conservation form
    dp/dz + d/dtau (-epsilon p^2 / 2) = 0 with the exact (Godunov) flux of the Riemann
    problem, MUSCL reconstruction with minmod slopes and Heun (SSP-RK2) steps in z. It is
    conservative and captures shocks without oscillations; truncating the result to K
    harmonics then removes the energy that a shock dissipates.

``"auto"`` (the default) uses ``"spectral"`` until the K-th harmonic exceeds a fraction
``switch_threshold`` of the fundamental anywhere on the plane, and ``"godunov"`` from then on.

Points where the pressure is below ``amplitude_threshold`` times the plane maximum are skipped:
there the nonlinear change is of relative order (threshold)^2.

With :class:`AdaptiveSmoothing`, the Godunov step also widens each shock to a set rise time
by adding artificial diffusivity where the shock is (Maxwell et al., *IEEE TUFFC* 64, 2017).
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.fft as sfft

from pywave3d._types import ComplexArray, IntArray, RealArray
from pywave3d.medium import Medium

NonlinearMethod = Literal["auto", "spectral", "godunov"]

#: 10-90 % rise time of the Taylor shock in units of rho0 delta / (beta dp): 4 artanh(0.8)
TAYLOR_RISE = 4 * math.atanh(0.8)
#: adaptive smoothing acts on fronts steeper than a Taylor front of rise GATE x tau_sh
GATE = 1.5


@dataclass(frozen=True, slots=True)
class AdaptiveSmoothing:
    """Artificial diffusivity that holds every shock at a set rise time.

    Maxwell et al. (*IEEE TUFFC* 64, 2017, eq. 5) add to the diffusivity, at each transverse
    point, the amount that makes the shock there a Taylor shock of rise time tau_sh. The steady
    (Taylor) solution of the Burgers equation, dp/dz = epsilon p dp/dtau
    + delta / (2 c0^3) d2p/dtau2, is a tanh front whose 10-90 % rise time is
    4 artanh(0.8) rho0 delta / (beta dp_sh) for a jump dp_sh, so

        delta + delta_a = tau_sh beta dp_sh / (4 artanh(0.8) rho0),   delta_a >= 0.

    (Eq. 5 of the paper, written in normalised variables, has the same constant:
    8.8 = 2 x 4 artanh(0.8).) The paper sets tau_sh to 7 steps of the time grid; here the time
    grid is that of the harmonics kept, T / (2 K), with K the number of harmonics.

    It acts in the Godunov step, where shocks are, and only at points whose front is steeper
    than a Taylor front of rise time 1.5 tau_sh with the same jump, so smooth waveforms are left
    alone (a sinusoid is, for at least 15 harmonics), while a front held at tau_sh stays
    smoothed. On the waveform after the step: the front is at the steepest rise (max dp/dtau),
    dp_sh is the rise from the minimum over the quarter period before it to the maximum over
    the quarter period after it. delta_a is then applied to the harmonics over the step,
    exp(-delta_a omega_n^2 dz / (2 c0^3)).

    Attributes:
        rise_points: tau_sh in steps T / (2 K) of the time grid.
    """

    rise_points: float = 7.0

    def __post_init__(self) -> None:
        if self.rise_points <= 0:
            raise ValueError("adaptive smoothing: rise_points must be positive")


def waveform(harmonics: ComplexArray, nt: int, workers: int = 1, axis: int = 0) -> RealArray:
    """Return the waveform p(tau_j), tau_j = j T / nt, from harmonics P_n, n = 1 .. K.

    ``p = Re sum_n P_n exp(+i n omega tau)``; the harmonics run along ``axis`` (0 or -1) and
    the samples replace them there. Harmonics above nt/2 are dropped.
    """
    h = np.moveaxis(harmonics, axis, 0)
    m = min(h.shape[0], nt // 2)
    spec = np.zeros((nt // 2 + 1, *h.shape[1:]), dtype=np.result_type(harmonics.dtype, np.complex64))
    np.multiply(h[:m], nt / 2, out=spec[1 : m + 1])
    spec = np.moveaxis(spec, 0, axis)
    return sfft.irfft(spec, n=nt, axis=axis, workers=workers, overwrite_x=True)


def harmonics_of(wave: RealArray, k: int, workers: int = 1, axis: int = 0) -> ComplexArray:
    """Return harmonics P_n, n = 1 .. k, of a waveform sampled over one period along ``axis``."""
    nt = wave.shape[axis]
    spec = np.moveaxis(sfft.rfft(wave, axis=axis, workers=workers), axis, 0)
    out = np.zeros((k, *spec.shape[1:]), dtype=spec.dtype)
    m = min(k, nt // 2)
    np.multiply(spec[1 : m + 1], 2 / nt, out=out[:m])
    return np.moveaxis(out, 0, axis)


def peak_pressures(
    harmonics: ComplexArray,
    workers: int = 1,
    iterations: int = 6,
    chunk: int = 8192,
    floor: tuple[RealArray | float, RealArray | float] | None = None,
) -> tuple[RealArray, RealArray]:
    """Peak positive and negative pressure (p+, p-) of the waveforms of harmonics (K, ...).

    The waveform is sampled at 4 K points per period. The largest local maximum of the samples
    (of -p for p-) is refined by Newton steps on the trigonometric series itself, kept within
    one sample of where it started, where a local maximum of samples must have its peak; so is
    the second largest wherever it could still come out on top. Sampling alone misses a peak
    between samples by up to 1 - cos(pi / (4 K)) of the fundamental (29% for K = 1, 2% for
    K = 4); refined, the error is at round-off for smooth waveforms. A refined value never falls
    below the sampled one, so steep (shocked) waveforms are at least as accurate as the samples.

    A sampled peak falls short of the true one by at most ``miss = (dphi^2 / 8) sum n^2 |P_n|``,
    dphi = 2 pi / (4 K). With ``floor = (f_plus, f_minus)`` (numbers, or arrays of the shape of
    the points), a point is refined only where its peak may exceed the floor: p+ where
    ``sampled p+ + miss > f_plus``, p- where ``|sampled p-| + miss > f_minus``. Elsewhere the
    sampled value is returned; it is then exact to within ``miss`` and known not to exceed
    the floor, which is all that a running maximum or a plane extreme needs.

    The points are taken ``chunk`` at a time, so the work arrays stay small whatever the plane.
    The samples are at the precision of ``harmonics`` (the FFT of a single-precision march stays
    single); the Newton steps are in double precision. Returns two float64 arrays of shape
    ``harmonics.shape[1:]``.
    """
    h = np.asarray(harmonics)
    k = h.shape[0]
    rest = h.shape[1:]
    flat = h.reshape(k, -1)
    npts = flat.shape[1]
    floors = [None, None] if floor is None else [np.broadcast_to(np.asarray(f, float), rest).ravel() for f in floor]
    pp = np.empty(npts)
    pm = np.empty(npts)
    step = max(1, chunk)
    for a in range(0, npts, step):
        part = flat[:, a : a + step]
        fl = [None if f is None else f[a : a + step] for f in floors]
        pp[a : a + step], pm[a : a + step] = _refined_extremes(part, workers, iterations, fl[0], fl[1])
    return pp.reshape(rest), pm.reshape(rest)


def _refined_extremes(
    flat: ComplexArray, workers: int, iterations: int, floor_plus: RealArray | None, floor_minus: RealArray | None
) -> tuple[RealArray, RealArray]:
    """(p+, p-) of the waveforms of harmonics ``flat`` (K, npts): see :func:`peak_pressures`."""
    k = flat.shape[0]
    nt = 4 * k
    wave = waveform(flat, nt, workers).astype(np.float64, copy=False)  # (nt, npts)
    flat = flat.astype(np.complex128, copy=False)
    n = np.arange(1, k + 1, dtype=float)
    # a sampled maximum is within half a sample of a peak of the waveform, so it falls short of
    # that peak by at most (sample / 2)^2 / 2 max|p''| <= (sample^2 / 8) sum n^2 |P_n|
    sample = 2 * np.pi / nt  # one sample, in omega tau
    miss = (sample**2 / 8) * ((n * n) @ np.abs(flat))
    plus = _extreme(flat, wave, miss, 1.0, floor_plus, iterations)
    minus = _extreme(flat, wave, miss, -1.0, floor_minus, iterations)
    return plus, minus


def _extreme(
    flat: ComplexArray, wave: RealArray, miss: RealArray, sign: float, floor: RealArray | None, iterations: int
) -> RealArray:
    """The largest of ``sign * p`` at each point (times ``sign``), refined where it may exceed
    ``floor``; ``wave`` holds the 4 K samples of each waveform (module-level helpers of
    :func:`peak_pressures`, so that beartype decorates them once and not at every call)."""
    nt, npts = wave.shape
    s = sign * wave
    j1 = np.argmax(s, axis=0)  # the largest sample: a local maximum
    best = s[j1, np.arange(npts)]  # the sampled peak: a lower bound, within miss of the true one
    todo = np.arange(npts) if floor is None else np.flatnonzero(best + miss > floor)
    if not todo.size:
        return sign * best
    best[todo] = _refine(flat[:, todo], j1[todo], s[:, todo], sign, iterations)
    if nt > 2:
        # the second largest local maximum of the samples can only win where its peak may rise
        # above the refined first (looked for at the refined points only)
        st = s[:, todo]
        cols = np.arange(todo.size)
        local = (st >= np.roll(st, 1, axis=0)) & (st >= np.roll(st, -1, axis=0))
        ranked = np.where(local, st, -np.inf)
        ranked[j1[todo], cols] = -np.inf
        j2 = np.argmax(ranked, axis=0)
        second = ranked[j2, cols]
        near = np.flatnonzero(np.isfinite(second) & (second + miss[todo] > best[todo]))
        if near.size:
            close = todo[near]
            again = _refine(flat[:, close], j2[near], s[:, close], sign, iterations)
            best[close] = np.maximum(best[close], again)
    return sign * best


def _refine(flat: ComplexArray, j: IntArray, s: RealArray, sign: float, iterations: int) -> RealArray:
    """Newton steps on ``sign * p`` from the samples ``j`` (one per point), each kept within one
    sample of where it started; never below that sample."""
    k = flat.shape[0]
    nt = s.shape[0]
    n = np.arange(1, k + 1, dtype=float)
    n2 = n * n
    sample = 2 * np.pi / nt
    phi0 = sample * j
    phi = phi0.copy()
    for _ in range(iterations):
        e = flat * _powers(phi, k)
        d1 = -(n @ e.imag)  # d/dphi of Re sum P_n e^{i n phi}
        d2 = -(n2 @ e.real)
        ok = sign * d2 < 0  # curving the right way: a maximum of sign * p
        step = np.where(ok, -d1 / np.where(d2 == 0, 1.0, d2), 0.0)
        step = np.clip(step, -sample / 2, sample / 2)
        phi = np.clip(phi + step, phi0 - sample, phi0 + sample)
        if np.abs(step).max(initial=0.0) < 1e-10 * sample:
            break
    value = sign * (flat * _powers(phi, k)).real.sum(axis=0)
    return np.maximum(value, s[j, np.arange(j.size)])


def _powers(phi: RealArray, k: int) -> ComplexArray:
    """exp(i n phi), n = 1 .. k, shape (k, npts): by a running product, not k exponentials."""
    return np.cumprod(np.broadcast_to(np.exp(1j * phi), (k, phi.size)), axis=0)


class NonlinearOperator:
    """Advances the harmonics of every transverse point by the nonlinear step.

    Args:
        medium: the medium (gives epsilon).
        frequency: base frequency [Hz].
        n_harmonics: K, the number of harmonics kept.
        method: ``"auto"``, ``"spectral"`` or ``"godunov"`` (see the module docstring).
        switch_threshold: ``"auto"`` switches to Godunov once max|P_K| / max|P_1| exceeds it.
        amplitude_threshold: points below this fraction of the plane maximum are skipped.
        godunov_oversample: the Godunov scheme works on ``2 K x godunov_oversample`` samples
            per period, which keeps its numerical diffusion away from the K harmonics kept.
        cfl: Courant number of the Godunov sub-steps (<= 1).
        rk_safety: the RK4 sub-step keeps dz epsilon omega K max|p| below this (< 2.8).
        workers: threads; the transverse points are shared between them.
        chunk_size: points per chunk of work.
        smoothing: optional adaptive artificial diffusivity at the shocks (Godunov only).
    """

    def __init__(
        self,
        medium: Medium,
        frequency: float,
        n_harmonics: int,
        *,
        method: NonlinearMethod = "auto",
        switch_threshold: float = 1e-3,
        amplitude_threshold: float = 1e-4,
        godunov_oversample: int = 4,
        cfl: float = 0.5,
        rk_safety: float = 1.0,
        workers: int = 1,
        chunk_size: int = 4096,
        smoothing: AdaptiveSmoothing | None = None,
    ) -> None:
        if method not in ("auto", "spectral", "godunov"):
            raise ValueError(f"nonlinear method must be 'auto', 'spectral' or 'godunov', not {method!r}")
        if n_harmonics < 1:
            raise ValueError("n_harmonics must be >= 1")
        if not 0 < cfl <= 1:
            raise ValueError("cfl must be in (0, 1]")
        self.epsilon = medium.epsilon
        self.omega = 2 * math.pi * frequency
        self.period = 1.0 / frequency
        self.k = n_harmonics
        self.method = method
        self.switch_threshold = switch_threshold
        self.amplitude_threshold = amplitude_threshold
        self.cfl = cfl
        self.rk_safety = rk_safety
        self.workers = workers
        self.n_godunov = 2 * n_harmonics * max(1, godunov_oversample)
        n = np.arange(1, n_harmonics + 1, dtype=float)
        self._coef = (0.5j * self.epsilon * self.omega) * n  # (K,)
        self.shocked = method == "godunov"
        self.last_substeps = 0
        self.chunk_size = chunk_size
        self._pool: ThreadPoolExecutor | None = None
        self.smoothing = smoothing
        self.diffusivity = medium.diffusivity
        # shock rise time tau_sh [s] and delta per unit jump, tau_sh beta / (4 artanh(0.8) rho0)
        self.rise_time = 0.0 if smoothing is None else smoothing.rise_points * self.period / (2 * n_harmonics)
        self._delta_per_pa = self.rise_time * medium.nonlinearity / (TAYLOR_RISE * medium.density)
        # absorption per unit diffusivity of each harmonic, omega_n^2 / (2 c0^3) [Np s/m^3]
        self._viscous = (self.omega * n) ** 2 / (2 * medium.sound_speed**3)
        #: largest artificial diffusivity of the last step [m^2/s] (0 without smoothing)
        self.last_artificial_diffusivity = 0.0

    # -- public --------------------------------------------------------------------------

    def choose(self, field: ComplexArray) -> Literal["spectral", "godunov"]:
        """Return the integrator that ``apply`` will use for ``field``."""
        if self.method != "auto":
            return self.method
        if not self.shocked and self.k > 1:
            top = float(np.abs(field[-1]).max())
            fundamental = float(np.abs(field[0]).max())
            if fundamental > 0 and top > self.switch_threshold * fundamental:
                self.shocked = True  # stays on: shocks, once formed, persist near the focus
        return "godunov" if self.shocked else "spectral"

    def apply(self, field: ComplexArray, dz: float, n_active: int | None = None) -> str:
        """Advance ``field`` (harmonics, shape (K, nx, ny)) in place over ``dz``.

        Args:
            field: the harmonics.
            dz: the step [m].
            n_active: harmonics above ``n_active`` are zero on entry (None: unknown). The
                spectral integrator then works on the first ``min(K, 4 n_active)`` only.

        Returns:
            The integrator used (``"spectral"`` or ``"godunov"``), or ``"none"`` if the field
            is zero or the medium linear.
        """
        if self.epsilon == 0:
            return "none"
        bound = np.abs(field).sum(axis=0)  # >= max_tau |p| at each point
        pmax = float(bound.max())
        if pmax == 0:
            return "none"
        method = self.choose(field)
        ka = self.k if n_active is None or method == "godunov" else min(self.k, 4 * max(1, n_active))
        mask = bound > self.amplitude_threshold * pmax
        if mask.all():
            q = np.ascontiguousarray(field[:ka].reshape(ka, -1).T)
            self._advance(q, dz, pmax, method)
            field[:ka] = q.T.reshape(ka, *field.shape[1:])
        else:
            q = np.ascontiguousarray(field[:ka, mask].T)
            self._advance(q, dz, pmax, method)
            field[:ka, mask] = q.T
        return method

    def close(self) -> None:
        """Shut down the worker threads (they are restarted when needed)."""
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    # -- integrators ---------------------------------------------------------------------

    def _advance(self, q: ComplexArray, dz: float, pmax: float, method: str) -> None:
        """Advance harmonics ``q`` of shape (npts, ka), in place, in chunks of points.

        Every point is independent, so the chunks (small enough to stay in cache) run on
        ``workers`` threads; numpy and the FFTs release the GIL. The number of sub-steps is
        set from the plane maximum, so the result does not depend on the chunking.
        """
        ka = q.shape[1]
        if method == "spectral":
            rate = self.epsilon * self.omega * ka * pmax * dz
            nsub = max(1, math.ceil(rate / self.rk_safety))
            m = sfft.next_fast_len(3 * ka + 1, real=True)
            coef = (self._coef[:ka] * (2 / m)).astype(q.dtype)  # at the march's precision

            def run(chunk: ComplexArray) -> None:
                chunk[...] = self._spectral(chunk, dz, nsub, m, coef)
        else:
            dtau = self.period / self.n_godunov
            nsub = max(1, math.ceil(self.epsilon * pmax * dz / (self.cfl * dtau)))

            peaks: list[float] = []

            def run(chunk: ComplexArray) -> None:
                chunk[...], top = self._godunov(chunk, dz, nsub)
                peaks.append(top)

        self.last_substeps = nsub
        size = self.chunk_size
        chunks = [q[i : i + size] for i in range(0, q.shape[0], size)]
        if self.workers > 1 and len(chunks) > 1:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(self.workers, thread_name_prefix="pywave3d-nl")
            list(self._pool.map(run, chunks))
        else:
            for c in chunks:
                run(c)
        if method != "spectral":
            self.last_artificial_diffusivity = max(peaks, default=0.0)

    @staticmethod
    def _rhs(p: ComplexArray, spec: ComplexArray, m: int, coef: ComplexArray) -> ComplexArray:
        """dP/dz for harmonics p of shape (npts, ka); ``spec`` is a zeroed (npts, m/2+1) buffer."""
        ka = p.shape[1]
        np.multiply(p, m / 2, out=spec[:, 1 : ka + 1])
        wave = sfft.irfft(spec, n=m, axis=-1)
        np.square(wave, out=wave)
        sq = sfft.rfft(wave, axis=-1)[:, 1 : ka + 1]
        sq *= coef
        return sq

    def _spectral(self, p: ComplexArray, dz: float, nsub: int, m: int, coef: ComplexArray) -> ComplexArray:
        """Classical RK4 over ``nsub`` sub-steps of dz / nsub."""
        h = dz / nsub
        spec = np.zeros((p.shape[0], m // 2 + 1), dtype=np.result_type(p.dtype, np.complex64))
        rhs = self._rhs
        for _ in range(nsub):
            k1 = rhs(p, spec, m, coef)
            k2 = rhs(p + (h / 2) * k1, spec, m, coef)
            acc = k1 + 2 * k2
            k3 = rhs(p + (h / 2) * k2, spec, m, coef)
            acc += 2 * k3
            acc += rhs(p + h * k3, spec, m, coef)
            acc *= h / 6
            p = p + acc.astype(p.dtype, copy=False)
        return p

    def _flux_divergence(self, u: RealArray, dtau: float) -> RealArray:
        """-(F_{j+1/2} - F_{j-1/2}) / dtau for f(u) = -epsilon u^2 / 2, periodic along axis -1.

        MUSCL reconstruction with minmod slopes, and the exact Godunov flux of this concave
        flux, F(uL, uR) = -(epsilon / 2) max(min(uL, 0)^2, max(uR, 0)^2).
        """
        fwd = np.empty_like(u)
        np.subtract(u[:, 1:], u[:, :-1], out=fwd[:, :-1])
        np.subtract(u[:, 0], u[:, -1], out=fwd[:, -1])
        bwd = np.roll(fwd, 1, axis=-1)
        # minmod(bwd, fwd)
        slope = np.where(bwd * fwd > 0, np.copysign(np.minimum(np.abs(bwd), np.abs(fwd)), fwd), 0.0)
        slope *= 0.5
        left = u + slope  # at j+1/2 from cell j
        right = u - slope  # at j-1/2 from cell j, i.e. at j+1/2 from cell j+1 after the roll
        right = np.roll(right, -1, axis=-1)
        np.minimum(left, 0.0, out=left)
        np.maximum(right, 0.0, out=right)
        np.square(left, out=left)
        np.square(right, out=right)
        flux = np.maximum(left, right, out=left)  # = -F * 2 / epsilon
        div = np.empty_like(u)
        np.subtract(flux[:, 1:], flux[:, :-1], out=div[:, 1:])
        np.subtract(flux[:, 0], flux[:, -1], out=div[:, 0])
        div *= 0.5 * self.epsilon / dtau
        return div

    def _artificial_diffusivity(self, u: RealArray, dtau: float) -> RealArray:
        """delta_a [m^2/s] at each point (row) of waveforms ``u`` of shape (npts, nt)."""
        npts, nt = u.shape
        rows = np.arange(npts)[:, None]
        rise = np.empty_like(u)
        np.subtract(np.roll(u, -1, axis=-1), u, out=rise)  # u[j+1] - u[j]
        steep = rise.argmax(axis=-1)
        slope = rise[rows[:, 0], steep] / dtau  # max dp/dtau
        # the waveform with its steepest rise in the middle
        c = nt // 2
        w = max(1, nt // 4)
        centred = u[rows, (steep[:, None] + 1 - c + np.arange(nt)[None, :]) % nt]
        jump = centred[:, c : c + w].max(axis=-1) - centred[:, c - w : c].min(axis=-1)
        # a Taylor front (dp/2) tanh(tau / T) of jump dp has the rise 2 artanh(0.8) T and the
        # largest slope dp / (2 T) = artanh(0.8) dp / rise. Fronts steeper than one of rise
        # GATE tau_sh are smoothed: GATE > 1 keeps a front held at tau_sh smoothed.
        steeper = slope * self.rise_time * GATE > math.atanh(0.8) * jump
        delta_a = np.maximum(self._delta_per_pa * jump - self.diffusivity, 0.0)
        return np.where(steeper & (jump > 0), delta_a, 0.0)

    def _godunov(self, p: ComplexArray, dz: float, nsub: int) -> tuple[ComplexArray, float]:
        """MUSCL-Godunov with Heun steps over ``nsub`` sub-steps, on the waveform.

        Returns the new harmonics and the largest artificial diffusivity applied (0 without
        smoothing).
        """
        nt = self.n_godunov
        u = waveform(p, nt, axis=-1)
        dtau = self.period / nt
        h = dz / nsub
        for _ in range(nsub):
            u1 = u + h * self._flux_divergence(u, dtau)
            u1 += h * self._flux_divergence(u1, dtau)
            u += u1
            u *= 0.5
        out = harmonics_of(u, p.shape[1], axis=-1).astype(p.dtype, copy=False)
        if self.smoothing is None:
            return out, 0.0
        delta_a = self._artificial_diffusivity(u, dtau)
        top = float(delta_a.max(initial=0.0))
        if top > 0:
            viscous = self._viscous[None, : p.shape[1]].astype(u.dtype)
            out *= np.exp(-dz * delta_a[:, None] * viscous).astype(out.real.dtype, copy=False)
        return out, top
