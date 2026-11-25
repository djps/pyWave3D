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
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import numpy as np
import scipy.fft as sfft

from pywave3d._types import ComplexArray, RealArray
from pywave3d.medium import Medium

NonlinearMethod = Literal["auto", "spectral", "godunov"]


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
            coef = self._coef[:ka] * (2 / m)

            def run(chunk: ComplexArray) -> None:
                chunk[...] = self._spectral(chunk, dz, nsub, m, coef)
        else:
            dtau = self.period / self.n_godunov
            nsub = max(1, math.ceil(self.epsilon * pmax * dz / (self.cfl * dtau)))

            def run(chunk: ComplexArray) -> None:
                chunk[...] = self._godunov(chunk, dz, nsub)

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

    def _godunov(self, p: ComplexArray, dz: float, nsub: int) -> ComplexArray:
        """MUSCL-Godunov with Heun steps over ``nsub`` sub-steps, on the waveform."""
        nt = self.n_godunov
        u = waveform(p, nt, axis=-1)
        dtau = self.period / nt
        h = dz / nsub
        for _ in range(nsub):
            u1 = u + h * self._flux_divergence(u, dtau)
            u1 += h * self._flux_divergence(u1, dtau)
            u += u1
            u *= 0.5
        return harmonics_of(u, p.shape[1], axis=-1).astype(p.dtype, copy=False)
