"""The computational grid: transverse (x, y) points, z planes and samples per period."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import scipy.fft as sfft

from pywave3d._types import RealArray


def _fast_even(n: int) -> int:
    """Smallest even m >= n whose prime factors are all 2, 3, 5, 7 or 11."""
    m = n + n % 2
    while sfft.next_fast_len(m) != m:
        m += 2
    return m


@dataclass(frozen=True)
class Grid:
    """A regular grid over [x0, x0 + lx] x [y0, y0 + ly] x [z0, z0 + lz].

    The end points are included (``dx = lx / (nx - 1)``) and nx and ny
    are even. The angular spectrum method treats the plane as periodic with period nx dx.
    Grids describing the output of other solvers (:mod:`pywave3d.preprocess`) may have odd
    sizes, ``ny = 1`` for a 2D field, and ``nz = 1``.

    Attributes:
        nx, ny: transverse points.
        nz: z planes, including z0 (so ``nz - 1`` steps of ``dz``).
        nt: samples per period of the waveform; harmonics 1 .. nt/2 are kept.
        dx, dy, dz: spacings [m].
        x0, y0, z0: the first point [m].
    """

    nx: int
    ny: int
    nz: int
    nt: int
    dx: float
    dy: float
    dz: float
    x0: float = 0.0
    y0: float = 0.0
    z0: float = 0.0

    def __post_init__(self) -> None:
        if self.nx < 2 or self.ny < 1 or self.nz < 1:
            raise ValueError("the grid needs at least 2 points in x and 1 in y and z")
        if self.nt < 2 or self.nt % 2:
            raise ValueError("nt (samples per period) must be even and >= 2")
        if min(self.dx, self.dy, self.dz) <= 0:
            raise ValueError("grid spacings must be positive")

    @classmethod
    def from_lengths(
        cls,
        lx: float,
        ly: float,
        lz: float,
        wavelength: float,
        points_per_wavelength: tuple[float, float, float] = (6.0, 6.0, 6.0),
        nt: int = 32,
        number_of_points: tuple[int, int, int] | None = None,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        fast_sizes: bool = True,
    ) -> Grid:
        """Build the grid from the domain lengths and the wavelength.

        Args:
            lx, ly, lz: domain lengths [m].
            wavelength: wavelength at the base frequency [m].
            points_per_wavelength: (x, y, z) points per wavelength, used when
                ``number_of_points`` is not given.
            nt: samples per period (rounded up to even).
            number_of_points: explicit (nx, ny, nz).
            origin: (x0, y0, z0) [m].
            fast_sizes: when nx and ny come from the points per wavelength, round them up to
                even sizes with small prime factors (e.g. 168 rather than 164 = 4 x 41), for
                which the FFTs are fastest.
        """
        if number_of_points is not None:
            nx, ny, nz = (int(v) for v in number_of_points)
        else:
            px, py, pz = points_per_wavelength
            nx = math.ceil(lx / (wavelength / px)) + 1
            ny = math.ceil(ly / (wavelength / py)) + 1
            nz = math.ceil(lz / (wavelength / pz)) + 1
            if fast_sizes:
                nx, ny = _fast_even(nx), _fast_even(ny)
        nx += nx % 2
        ny += ny % 2
        nt += nt % 2
        return cls(nx, ny, nz, nt, lx / (nx - 1), ly / (ny - 1), lz / (nz - 1), *origin)

    # -- coordinates ---------------------------------------------------------------------

    @cached_property
    def x(self) -> RealArray:
        """x coordinates [m], shape (nx,)."""
        return self.x0 + self.dx * np.arange(self.nx)

    @cached_property
    def y(self) -> RealArray:
        """y coordinates [m], shape (ny,)."""
        return self.y0 + self.dy * np.arange(self.ny)

    @cached_property
    def z(self) -> RealArray:
        """z planes [m], shape (nz,)."""
        return self.z0 + self.dz * np.arange(self.nz)

    @cached_property
    def kx(self) -> RealArray:
        """Angular wavenumbers in x [rad/m], in FFT order, shape (nx,)."""
        return 2 * np.pi * np.fft.fftfreq(self.nx, self.dx)

    @cached_property
    def ky(self) -> RealArray:
        """Angular wavenumbers in y [rad/m], in FFT order, shape (ny,)."""
        return 2 * np.pi * np.fft.fftfreq(self.ny, self.dy)

    @property
    def n_harmonics(self) -> int:
        """Number of harmonics kept, K = nt / 2."""
        return self.nt // 2

    @property
    def n_steps(self) -> int:
        """Number of steps of dz, nz - 1."""
        return self.nz - 1

    @property
    def centre_index(self) -> tuple[int, int]:
        """Indices (nx/2, ny/2) of the beam axis, the plane centre."""
        return self.nx // 2, self.ny // 2

    @property
    def centre(self) -> tuple[float, float]:
        """Coordinates of the beam axis [m]."""
        i, j = self.centre_index
        return float(self.x[i]), float(self.y[j])

    @property
    def lengths(self) -> tuple[float, float, float]:
        """Domain lengths (lx, ly, lz) [m]."""
        return (self.nx - 1) * self.dx, (self.ny - 1) * self.dy, (self.nz - 1) * self.dz

    def nearest(self, x: float, y: float, z: float | None = None) -> tuple[int, ...]:
        """Indices of the grid point nearest to (x, y) or (x, y, z), clamped to the grid."""
        i = int(np.clip(round((x - self.x0) / self.dx), 0, self.nx - 1))
        j = int(np.clip(round((y - self.y0) / self.dy), 0, self.ny - 1))
        if z is None:
            return i, j
        k = int(np.clip(round((z - self.z0) / self.dz), 0, self.nz - 1))
        return i, j, k

    def summary(self) -> str:
        """One-line description for the log."""
        return (
            f"{self.nx} x {self.ny} points (dx = {self.dx:.4e} m, dy = {self.dy:.4e} m), "
            f"{self.n_steps} steps of dz = {self.dz:.4e} m, {self.nt} samples per period "
            f"({self.n_harmonics} harmonics)"
        )
