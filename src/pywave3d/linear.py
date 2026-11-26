"""The linear part of a step: diffraction and absorption, exact per harmonic.

In the retarded time tau = t - z/c0 with the convention p = Re sum P_n exp(+i n omega tau),
a plane-wave component (kx, ky) of harmonic n, with k_n = n omega / c0, evolves over dz as

    P_n(kx, ky, z + dz) = P_n(kx, ky, z) exp(-i (kz_n - k_n) dz - (alpha_n + i dk_n) dz),

    kz_n = sqrt(k_n^2 - kx^2 - ky^2)          (propagating),
    kz_n = -i sqrt(kx^2 + ky^2 - k_n^2)       (evanescent: decays),

where alpha_n is the absorption and dk_n the Kramers-Kronig dispersion of the medium. This is
the exact one-way (angular spectrum) propagator; ``paraxial=True`` replaces kz_n - k_n by
-(kx^2 + ky^2) / (2 k_n), the KZK (parabolic) approximation. Absorption does not depend on
(x, y), so it commutes with diffraction and both are applied with one pair of 2D FFTs.

The transverse plane is periodic. Against wrap-around an :class:`AbsorbingLayer` can be placed
inside the lateral edges, either as a sponge (every component attenuated by exp(-sigma dz)) or
as a perfectly matched layer. The PML is the complex coordinate stretch
d/dx -> (1 - i sigma_x(x) / k_n)^-1 d/dx (the sign suits exp(+i omega t)), under which a plane
wave travelling at kx/k_n crosses the layer with the decay sigma_x |kx| / k_n per unit x, so
per unit z it decays at

    gamma_n = sigma_x(x) kx^2 / (k_n kz_n) + sigma_y(y) ky^2 / (k_n kz_n)

(kz_n -> k_n in the paraxial case). Unlike the sponge it acts on the lateral component of the
motion only: a wave heading out through the side is absorbed, a wave running along z next to
the edge is almost untouched. Because gamma depends on both position and wavenumber, the
symbol exp(-gamma dz) is applied at ``levels`` reference values of sigma and the fields are
interpolated linearly in sigma at each point (phase shift plus interpolation, Gazdag and
Sguazzero 1984). Each level is a contraction and the weights are convex, so the step is stable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.fft as sfft

from pywave3d._types import ComplexArray, RealArray
from pywave3d.grid import Grid
from pywave3d.medium import Medium

LAYER_KINDS = ("sponge", "pml")


@dataclass(frozen=True, slots=True)
class AbsorbingLayer:
    """A lossy layer along the edges of the (x, y) plane, against periodic wrap-around.

    Inside the outer ``width`` of each side, sigma = strength * (d / width)^power, d being the
    distance into the layer; the layer lies inside the domain.

    * ``kind="sponge"``: every harmonic and direction is attenuated by exp(-sigma dz) per step;
    * ``kind="pml"``: the perfectly matched layer of the module docstring. A wave travelling
      at the angle theta from z decays by exp(-sigma sin(theta) dx) as it crosses the layer; a
      reflection of R at normal incidence through the full layer and back needs
      ``strength = (power + 1) ln(1/R) / (2 width)`` (:meth:`for_reflection`).

    Attributes:
        width: layer thickness on each side [m].
        strength: sigma at the outer edge [Np/m].
        power: profile exponent.
        kind: ``"sponge"`` or ``"pml"``.
        levels: PML only: number of reference values of sigma (more is more accurate and
            costs one inverse FFT per level, direction and harmonic).
    """

    width: float = 0.0
    strength: float = 0.0
    power: float = 2.0
    kind: str = "sponge"
    levels: int = 4

    def __post_init__(self) -> None:
        if self.kind not in LAYER_KINDS:
            raise ValueError(f"absorbing layer: kind must be one of {LAYER_KINDS}, not {self.kind!r}")
        if self.levels < 1:
            raise ValueError("absorbing layer: levels must be at least 1")

    @classmethod
    def for_reflection(
        cls, width: float, reflection: float = 1e-4, power: float = 2.0, kind: str = "pml", levels: int = 4
    ) -> AbsorbingLayer:
        """A layer whose round trip at normal incidence attenuates by ``reflection``."""
        if not 0 < reflection < 1:
            raise ValueError("absorbing layer: reflection must be in (0, 1)")
        strength = (power + 1) * np.log(1 / reflection) / (2 * width)
        return cls(width, float(strength), power, kind, levels)

    @property
    def active(self) -> bool:
        """True if the layer does anything."""
        return self.width > 0 and self.strength > 0

    def _depth(self, c: RealArray, start: float, length: float) -> RealArray:
        d = np.maximum(self.width - (c - start), self.width - (start + length - c))
        return np.clip(d, 0.0, None) / self.width

    def profiles(self, grid: Grid) -> tuple[RealArray, RealArray] | None:
        """Return (sigma_x(x), sigma_y(y)) [Np/m], or None for no layer."""
        if not self.active:
            return None
        lx, ly, _ = grid.lengths
        return (
            self.strength * self._depth(grid.x, grid.x0, lx) ** self.power,
            self.strength * self._depth(grid.y, grid.y0, ly) ** self.power,
        )

    def profile(self, grid: Grid) -> RealArray | None:
        """Return the sponge's sigma(x, y) [Np/m] with shape (nx, ny), or None for no layer."""
        sig = self.profiles(grid)
        if sig is None:
            return None
        return np.maximum(sig[0][:, None], sig[1][None, :])


def _logcosh(x: float) -> float:
    """log(cosh(x)), without overflow."""
    a = abs(x)
    return a + math.log1p(math.exp(-2 * a)) - math.log(2.0)


@dataclass(frozen=True, slots=True)
class AxialSmoothing:
    """Artificial absorption around the focus against the oscillations that shocks cause.

    Yuldashev and Khokhlova (*Acoust. Phys.* 57, 2011, eq. 11) multiply the absorption of
    water by an axial profile that rises to ``A`` around the focus and falls to ``B`` after it:

        m(z) = a1 + (A - 1)/2 [tanh((z - z1)/dz1) - a2 tanh((z - z2)/dz2)],
        a1 = (1 + B)/2,  a2 = 1 + (1 - B)/(A - 1),

    so m = 1 before z1, A between z1 and z2 and B after z2. (The manuscript prints
    a2 = 1 + (1 - B)(A - 1), which does not give these limits.) It is evaluated here in the
    equivalent form, which is also defined at A = 1,

        m(z) = 1 + (A - 1) s((z - z1)/dz1) - (A - B) s((z - z2)/dz2),  s(x) = (1 + tanh x)/2.

    The multiplier scales the thermoviscous part of the absorption: the diffusivity delta
    becomes m(z) delta, as in the adaptive scheme of Maxwell et al. (*IEEE TUFFC* 64, 2017),
    so each harmonic n gains the extra absorption (m - 1) delta omega_n^2 / (2 c0^3). Over a
    step the profile is integrated exactly, so steps longer than the transition widths are
    handled correctly. z is measured from the source plane.

    The defaults of the geometry are those of the paper: z1 = 0.96 F, z2 = 1.03 F and
    dz1 = dz2 = 0.01 F. The paper chose A for each source level (2.5, 5, 20 and 40 as the
    source intensity rose from 2 to 10 W/cm^2, about 7 points per shock) and gives no value for
    B, so the defaults A = 5 and B = 1 (back to the physical absorption after the focus) are
    starting points to adjust, not values from the paper.

    Attributes:
        focal_length: F [m], the distance from the source plane to the focus; 0 takes the
            transducer's focusing depth (see :class:`pywave3d.WaveSolver`).
        peak_multiplier: A, the multiplier between z1 and z2.
        post_focal_multiplier: B, the multiplier after z2.
        start, end: z1 / F and z2 / F.
        start_width, end_width: dz1 / F and dz2 / F, the widths of the transitions.
    """

    focal_length: float = 0.0
    peak_multiplier: float = 5.0
    post_focal_multiplier: float = 1.0
    start: float = 0.96
    end: float = 1.03
    start_width: float = 0.01
    end_width: float = 0.01

    def __post_init__(self) -> None:
        if self.focal_length < 0:
            raise ValueError("shock smoothing: focal_length must be >= 0")
        if self.peak_multiplier < 0 or self.post_focal_multiplier < 0:
            raise ValueError("shock smoothing: the multipliers must be >= 0")
        if self.start_width <= 0 or self.end_width <= 0:
            raise ValueError("shock smoothing: the transition widths must be positive")
        if self.start >= self.end:
            raise ValueError("shock smoothing: start must be before end")

    def _steps(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """(height, centre [m], width [m]) of the rising and the falling step."""
        f = self.focal_length
        if f <= 0:
            raise ValueError("shock smoothing: the focal length is not set")
        a, b = self.peak_multiplier, self.post_focal_multiplier
        return (a - 1, self.start * f, self.start_width * f), (b - a, self.end * f, self.end_width * f)

    def multiplier(self, z: RealArray | float) -> RealArray:
        """m(z) at distances ``z`` from the source plane [m]."""
        z = np.asarray(z, dtype=float)
        m = np.ones_like(z)
        for height, centre, width in self._steps():
            m = m + height * 0.5 * (1 + np.tanh((z - centre) / width))
        return np.asarray(m)

    def excess(self, za: float, zb: float) -> float:
        """The integral of m(z) - 1 from ``za`` to ``zb`` [m] (distances from the source plane)."""
        total = 0.0
        for height, centre, width in self._steps():
            # integral of s((z - c)/w) = (z - c)/2 + (w/2) log cosh((z - c)/w)
            ia = (za - centre) / 2 + width / 2 * _logcosh((za - centre) / width)
            ib = (zb - centre) / 2 + width / 2 * _logcosh((zb - centre) / width)
            total += height * (ib - ia)
        return total


class _PML:
    """The PML symbol at reference levels of sigma, for one step fraction.

    For direction d (x or y) and harmonic n, level j has sigma_j = j * strength / levels and
    multiplies the spectrum by ``decay_d ** j``, decay_d = exp(-f dz (strength/levels) q_d),
    q_x = kx^2 / (k_n kz_n). At a point with sigma_d between two levels the two fields are
    blended linearly.
    """

    def __init__(
        self, q: list[RealArray], sigma: tuple[RealArray, RealArray], layer: AbsorbingLayer, dz: float
    ) -> None:
        step = layer.strength / layer.levels
        self.decay = [np.exp(-dz * step * qd) for qd in q]  # each (K, nx, ny)
        self.levels = layer.levels
        # interpolation weights: (levels + 1, nx, 1) for x and (levels + 1, 1, ny) for y
        self.weights = []
        self.rows = []
        for d, sig in enumerate(sigma):
            pos = sig / step  # fractional level
            j = np.arange(self.levels + 1)[:, None]
            w = np.clip(1 - np.abs(pos[None, :] - j), 0.0, None)
            shape = (self.levels + 1, -1, 1) if d == 0 else (self.levels + 1, 1, -1)
            self.weights.append(w.reshape(shape))
            self.rows.append(np.flatnonzero(sig > 0))


class LinearPropagator:
    """Applies diffraction + absorption (+ dispersion) over a step, or half a step.

    The operators for dz and dz/2 are precomputed for all K harmonics, so a step costs one
    forward and one inverse 2D FFT per active harmonic and a multiplication.

    Args:
        grid: the grid.
        medium: the medium.
        frequency: base frequency [Hz].
        diffraction: include diffraction (False leaves only absorption, applied without FFTs).
        absorption: include absorption and dispersion.
        paraxial: use the parabolic (KZK) approximation of the propagator.
        layer: optional absorbing boundary layer (sponge or PML). The PML needs diffraction:
            without it nothing moves sideways and the layer does nothing.
        dtype: complex dtype of the field.
        workers: threads for the FFTs.
        smoothing: optional artificial absorption around the focus (:class:`AxialSmoothing`,
            with its focal length set). It is applied whether or not ``absorption`` is on, and
            needs the position of each step: see :meth:`apply`.
    """

    def __init__(
        self,
        grid: Grid,
        medium: Medium,
        frequency: float,
        *,
        diffraction: bool = True,
        absorption: bool = True,
        paraxial: bool = False,
        layer: AbsorbingLayer | None = None,
        dtype: type[np.complexfloating] = np.complex128,
        workers: int = 1,
        smoothing: AxialSmoothing | None = None,
    ) -> None:
        self.grid = grid
        self.diffraction = diffraction
        self.workers = workers
        n = np.arange(1, grid.n_harmonics + 1, dtype=float)
        kn = 2 * np.pi * frequency * n / medium.sound_speed  # (K,)
        if smoothing is not None:
            smoothing._steps()  # check that the focal length is set
        self.smoothing = smoothing
        # thermoviscous absorption per unit multiplier: delta omega_n^2 / (2 c0^3) [Np/m]
        omega = 2 * np.pi * frequency * n
        self._viscous = medium.diffusivity * omega**2 / (2 * medium.sound_speed**3)

        loss = np.zeros_like(n, dtype=complex)
        if absorption:
            loss = medium.alpha(frequency, n) + 1j * medium.dispersion(n)  # (K,)
        self._loss = loss

        self._pml_full: _PML | None = None
        self._pml_half: _PML | None = None
        self._layer_full: RealArray | None = None
        self._layer_half: RealArray | None = None
        real = np.finfo(dtype).dtype

        if diffraction:
            kt2 = (grid.kx[:, None] ** 2 + grid.ky[None, :] ** 2)[None, :, :]  # (1, nx, ny)
            k2 = kn[:, None, None] ** 2
            if layer is not None and layer.kind == "pml" and layer.active:
                sigma = layer.profiles(grid)
                assert sigma is not None
                # kz_n, kept away from 0 so that grazing and evanescent components are
                # absorbed strongly but finitely
                kz = kn[:, None, None] * np.ones_like(kt2) if paraxial else np.sqrt(np.clip(k2 - kt2, 0.0, None))
                kz = np.maximum(kz, 1e-2 * kn[:, None, None])
                q = [
                    (grid.kx[None, :, None] ** 2 / (kn[:, None, None] * kz)).astype(real),
                    (grid.ky[None, None, :] ** 2 / (kn[:, None, None] * kz)).astype(real),
                ]
                self._pml_full = _PML(q, sigma, layer, grid.dz)
                self._pml_half = _PML(q, sigma, layer, grid.dz / 2)
            if paraxial:
                phase = -kt2 / (2 * kn[:, None, None])  # kz - k
                exponent = -1j * phase
            else:
                kz = np.sqrt((k2 - kt2).astype(complex))
                # propagating: kz real >= 0; evanescent: kz = +i |.| -> exp(-|.| dz) decays
                exponent = -1j * (kz - kn[:, None, None])
                exponent = np.where(k2 >= kt2, exponent, -np.abs(kz) + 1j * kn[:, None, None])
            exponent = exponent - loss[:, None, None]
            self._full = np.exp(exponent * grid.dz).astype(dtype)
            self._half = np.exp(exponent * (grid.dz / 2)).astype(dtype)
        else:
            self._full = np.exp(-loss * grid.dz).astype(dtype)[:, None, None]
            self._half = np.exp(-loss * grid.dz / 2).astype(dtype)[:, None, None]

        if layer is not None and layer.kind == "sponge":
            sponge = layer.profile(grid)
            if sponge is not None:
                self._layer_full = np.exp(-sponge * grid.dz).astype(real)
                self._layer_half = np.exp(-sponge * grid.dz / 2).astype(real)

    def apply(
        self, field: ComplexArray, fraction: float = 1.0, n_active: int | None = None, z: float | None = None
    ) -> None:
        """Advance ``field`` (shape (K, nx, ny)) in place by ``fraction`` x dz.

        Args:
            field: the harmonics P_n(x, y), n = 1 .. K.
            fraction: 1.0 for a full step, 0.5 for half a step.
            n_active: only the first ``n_active`` harmonics are transformed (the others must be
                zero); None for all.
            z: where the step starts [m, grid coordinate]; needed with shock smoothing, which
                is integrated from z to z + fraction dz.
        """
        if fraction == 1.0:
            op, layer, pml = self._full, self._layer_full, self._pml_full
        elif fraction == 0.5:
            op, layer, pml = self._half, self._layer_half, self._pml_half
        else:
            raise ValueError("fraction must be 1.0 or 0.5")
        na = field.shape[0] if n_active is None else n_active
        if na == 0:
            return
        sub = field[:na]
        if self.diffraction:
            spec = sfft.fft2(sub, axes=(1, 2), workers=self.workers, overwrite_x=False)
            spec *= op[:na]
            sub[...] = sfft.ifft2(spec, axes=(1, 2), workers=self.workers, overwrite_x=pml is None)
            if pml is not None:
                self._apply_pml(sub, spec, pml, na)
        else:
            sub *= op[:na]
        if layer is not None:
            sub *= layer
        if self.smoothing is not None:
            if z is None:
                raise ValueError("shock smoothing needs the position z of the step")
            za = z - self.grid.z0
            excess = self.smoothing.excess(za, za + fraction * self.grid.dz)
            if excess != 0.0:
                sub *= np.exp(-self._viscous[:na] * excess).astype(sub.real.dtype)[:, None, None]

    def _apply_pml(self, sub: ComplexArray, spec: ComplexArray, pml: _PML, na: int) -> None:
        """Damp ``sub`` (already propagated; ``spec`` is its spectrum) in the PML, x then y."""
        for d in range(2):
            rows = pml.rows[d]
            if rows.size == 0:
                continue
            if d == 1:  # y after x: the spectrum of the x-damped field
                spec = sfft.fft2(sub, axes=(1, 2), workers=self.workers)
            w = pml.weights[d]
            take = (slice(None), rows, slice(None)) if d == 0 else (slice(None), slice(None), rows)
            wsel = w[(slice(None), rows, slice(None))] if d == 0 else w[(slice(None), slice(None), rows)]
            out = sub[take] * wsel[0]
            level = spec.copy()
            for j in range(1, pml.levels + 1):
                level *= pml.decay[d][:na]
                field = sfft.ifft2(level, axes=(1, 2), workers=self.workers)
                out += field[take] * wsel[j]
            sub[take] = out
