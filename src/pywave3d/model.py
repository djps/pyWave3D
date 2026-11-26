"""The physical model: medium, geometry, transducer and the source-plane field.

:class:`WaveModel` reads the ``Model`` section of the options file with these keys::

    "Model": {
      "Geometry": {"lx": 0.04, "ly": 0.04, "lz": 0.10, "origin": [0, 0, 0]},
      "PhysicalProperties": {"BaseFrequency": 1e6, "SpeedofSound": 1482, "Density": 998,
                             "NonlinearityCoefficient": 3.5, "Diffusivity": 4.33e-6,
                             "Attenuation": {"type": "power_law", "pre_factor": 0.0,
                                             "power": 2.0, "dispersion": false}},
      "Transducer": {"outer_radius": 0.0125, "inner_radius": 0, "focusing_depth": 0.05,
                     "frequency": 1e6, "power": 50},
      "Initial": {"InputFile": "pxy_at_z0.txt"},
      "AbsorbingBC": {"type": "pml", "width": 0.004, "reflection": 1e-4, "power": 2, "levels": 4}
    }

Every key is optional; missing ones take the water/focused-transducer defaults below. Without
``Initial.InputFile`` the source is the transducer (a uniform focused piston). A text input has
lines ``x y Re(p) Im(p)`` (``#`` starts a comment) holding the first harmonic; a binary input
(``"type": "binary"``) has raw x and y coordinate files and one raw real and one raw imaginary
file per harmonic, column-major (Fortran order).

The phase convention is p = Re sum P_n exp(+i n omega t): a source converging on
z = d has the phase exp(+i k (sqrt(r^2 + d^2) - d)).

``AbsorbingBC`` (used when ``Solver.Options.AbsorbingBC`` is true) takes ``type`` (``"sponge"``,
the default, or ``"pml"``), ``width`` [m], ``power`` and either ``strength`` (sigma at the edge,
Np/m) or ``reflection`` (the round-trip attenuation at normal incidence, from which the strength
is computed), and for the PML ``levels`` (4); see :class:`pywave3d.linear.AbsorbingLayer`.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator, griddata

from pywave3d._types import ComplexArray, RealArray
from pywave3d.grid import Grid
from pywave3d.linear import AbsorbingLayer
from pywave3d.medium import Attenuation, Medium
from pywave3d.options import Options

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Transducer:
    """A single-element focused piston (a spherical cap seen on its plane).

    Attributes:
        outer_radius, inner_radius: aperture radii [m] (inner > 0: a central opening).
        focusing_depth: geometric focus [m]; 0 for an unfocused piston.
        power: acoustic power [W]; the uniform amplitude is sqrt(2 rho c P / (pi (a^2 - b^2))).
        amplitude: if > 0, the source amplitude [Pa], overriding ``power``.
    """

    outer_radius: float = 0.0125
    inner_radius: float = 0.0
    focusing_depth: float = 0.050
    power: float = 50.0
    amplitude: float = 0.0

    def __post_init__(self) -> None:
        if not 0 <= self.inner_radius < self.outer_radius:
            raise ValueError("need 0 <= inner_radius < outer_radius")
        if self.focusing_depth < 0 or self.power < 0 or self.amplitude < 0:
            raise ValueError("focusing_depth, power and amplitude must be >= 0")
        if 0 < self.focusing_depth <= self.outer_radius:
            raise ValueError("focusing_depth must exceed outer_radius")

    def source_amplitude(self, medium: Medium) -> float:
        """Uniform pressure amplitude on the aperture [Pa]."""
        if self.amplitude > 0:
            return self.amplitude
        area = math.pi * (self.outer_radius**2 - self.inner_radius**2)
        return math.sqrt(2 * medium.impedance * self.power / area)

    def field(self, grid: Grid, medium: Medium, frequency: float, centre: tuple[float, float]) -> ComplexArray:
        """First harmonic on the grid, shape (nx, ny), centred on ``centre``.

        The amplitude is uniform; points on the rim get the fraction of their cell inside
        the aperture (estimated on a 4 x 4 sub-grid), which reduces the staircase error.
        """
        k = 2 * math.pi * frequency / medium.sound_speed
        p0 = self.source_amplitude(medium)
        x = grid.x - centre[0]
        y = grid.y - centre[1]
        # area fraction of each cell inside the annulus
        s = (np.arange(4) + 0.5) / 4 - 0.5
        xs = (x[:, None] + s[None, :] * grid.dx)[:, None, :, None]
        ys = (y[:, None] + s[None, :] * grid.dy)[None, :, None, :]
        rs = np.hypot(xs, ys)
        weight = ((rs <= self.outer_radius) & (rs >= self.inner_radius)).mean(axis=(2, 3))
        r = np.hypot(x[:, None], y[None, :])
        if self.focusing_depth > 0:
            d = self.focusing_depth
            phase = k * (np.sqrt(r**2 + d**2) - d)
        else:
            phase = np.zeros_like(r)
        return p0 * weight * np.exp(1j * phase)


@dataclass(frozen=True, slots=True)
class Source:
    """Harmonics of the pressure on a regular (x, y) grid, as read from a file.

    Attributes:
        x, y: coordinates of the data [m], increasing.
        harmonics: complex amplitudes [Pa], shape (nh, len(x), len(y)).
    """

    x: RealArray
    y: RealArray
    harmonics: ComplexArray

    def __post_init__(self) -> None:
        if self.harmonics.ndim != 3 or self.harmonics.shape[1:] != (self.x.size, self.y.size):
            raise ValueError("harmonics must have shape (nh, len(x), len(y))")

    @classmethod
    def from_text(cls, path: str | Path) -> Source:
        """Read ``x y Re(p) Im(p)`` lines (first harmonic) on a regular grid, in any order."""
        data = np.loadtxt(path, comments="#", ndmin=2)
        if data.shape[1] < 4 or data.shape[0] == 0:
            raise ValueError(f"{path}: expected lines of 'x y real imag'")
        xs, ix = np.unique(data[:, 0], return_inverse=True)
        ys, iy = np.unique(data[:, 1], return_inverse=True)
        values = data[:, 2] + 1j * data[:, 3]
        if xs.size * ys.size == data.shape[0] and xs.size > 1 and ys.size > 1:
            grid = np.zeros((1, xs.size, ys.size), dtype=complex)
            grid[0, ix, iy] = values
            return cls(xs, ys, grid)
        # scattered points: interpolate onto a regular grid of the same extent
        nx = max(2, round(math.sqrt(data.shape[0])))
        gx = np.linspace(xs[0], xs[-1], nx)
        gy = np.linspace(ys[0], ys[-1], nx)
        gxx, gyy = np.meshgrid(gx, gy, indexing="ij")
        grid = griddata(data[:, :2], values, (gxx, gyy), method="linear", fill_value=0.0)
        return cls(gx, gy, grid[None])

    @classmethod
    def from_binary(
        cls,
        x_file: str | Path,
        y_file: str | Path,
        real_files: list[Path],
        imag_files: list[Path],
        dtype: type[np.floating] = np.float64,
    ) -> Source:
        """Read the binary input: text coordinate files and raw column-major harmonics."""
        x = np.loadtxt(x_file, ndmin=1).astype(float)
        y = np.loadtxt(y_file, ndmin=1).astype(float)
        if len(real_files) != len(imag_files) or not real_files:
            raise ValueError("binary input needs as many real as imaginary files (at least one)")
        harmonics = np.empty((len(real_files), x.size, y.size), dtype=complex)
        for n, (rf, jf) in enumerate(zip(real_files, imag_files, strict=True)):
            re = np.fromfile(rf, dtype=dtype)
            im = np.fromfile(jf, dtype=dtype)
            if re.size != x.size * y.size or im.size != x.size * y.size:
                raise ValueError(f"harmonic {n + 1}: expected {x.size * y.size} values in {rf} and {jf}")
            harmonics[n] = (re + 1j * im).reshape(x.size, y.size, order="F")
        return cls(x, y, harmonics)

    def on_grid(self, grid: Grid, n_harmonics: int) -> ComplexArray:
        """Bilinear interpolation onto the grid, zero outside the data; shape (K, nx, ny)."""
        out = np.zeros((n_harmonics, grid.nx, grid.ny), dtype=complex)
        # snap points within round-off of the data edges back inside
        tx = 1e-6 * (self.x[-1] - self.x[0])
        ty = 1e-6 * (self.y[-1] - self.y[0])
        gx = np.where(np.abs(grid.x - self.x[0]) < tx, self.x[0], grid.x)
        gx = np.where(np.abs(gx - self.x[-1]) < tx, self.x[-1], gx)
        gy = np.where(np.abs(grid.y - self.y[0]) < ty, self.y[0], grid.y)
        gy = np.where(np.abs(gy - self.y[-1]) < ty, self.y[-1], gy)
        pts = np.stack(np.meshgrid(gx, gy, indexing="ij"), axis=-1)
        for n in range(min(n_harmonics, self.harmonics.shape[0])):
            interp = RegularGridInterpolator(
                (self.x, self.y), self.harmonics[n], method="linear", bounds_error=False, fill_value=0.0
            )
            out[n] = interp(pts)
        if self.harmonics.shape[0] > n_harmonics:
            log.warning("source has %d harmonics, only %d are kept", self.harmonics.shape[0], n_harmonics)
        return out


@dataclass
class WaveModel:
    """Medium, geometry, transducer and source of a simulation.

    Attributes:
        medium: the medium (water by default).
        frequency: base frequency [Hz].
        lengths: domain lengths (lx, ly, lz) [m].
        origin: domain origin (x0, y0, z0) [m].
        transducer: the transducer (used as the source when ``source`` is None, and for the
            focal gain and shift of the metrics).
        source: source-plane harmonics read from a file, or None.
        layer: absorbing boundary layer (used when the solver enables it).
    """

    medium: Medium = field(default_factory=Medium)
    frequency: float = 1.0e6
    lengths: tuple[float, float, float] = (0.040, 0.040, 0.100)
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    transducer: Transducer = field(default_factory=Transducer)
    source: Source | None = None
    layer: AbsorbingLayer = field(default_factory=AbsorbingLayer)

    def __post_init__(self) -> None:
        if self.frequency <= 0:
            raise ValueError("frequency must be positive")
        if min(self.lengths) <= 0:
            raise ValueError("domain lengths must be positive")

    @property
    def wavelength(self) -> float:
        """Wavelength at the base frequency [m]."""
        return self.medium.sound_speed / self.frequency

    @classmethod
    def from_options(cls, opts: Options) -> WaveModel:
        """Build the model from the ``Model`` node of the options (see the module docstring)."""
        d = cls()
        geo = opts.child("Geometry")
        lengths = (
            geo.get("lx", float, d.lengths[0]),
            geo.get("ly", float, d.lengths[1]),
            geo.get("lz", float, d.lengths[2]),
        )
        origin = geo.get("origin", list[float], list(d.origin))
        if len(origin) != 3:
            raise TypeError(f"options: '{geo.path}.origin' should be three numbers (x, y, z)")

        phy = opts.child("PhysicalProperties")
        dm = d.medium
        att = phy.child("Attenuation")
        law = att.get("type", str, "thermoviscous").lower()
        if law in ("standard", "quadratic", "thermoviscous", ""):
            law = "thermoviscous"
        attenuation = Attenuation(
            law=law,  # type: ignore[arg-type]
            alpha0=att.get("pre_factor", float, 0.0),
            power=att.get("power", float, 2.0),
            dispersion=att.get("dispersion", bool, False),
        )
        medium = Medium(
            sound_speed=phy.get("SpeedofSound", float, dm.sound_speed),
            density=phy.get("Density", float, dm.density),
            nonlinearity=phy.get("NonlinearityCoefficient", float, dm.nonlinearity),
            diffusivity=phy.get("Diffusivity", float, dm.diffusivity),
            attenuation=attenuation,
        )
        frequency = phy.get("BaseFrequency", float, d.frequency)

        tr = opts.child("Transducer")
        dt = d.transducer
        transducer = Transducer(
            outer_radius=tr.get("outer_radius", float, dt.outer_radius),
            inner_radius=tr.get("inner_radius", float, dt.inner_radius),
            focusing_depth=tr.get("focusing_depth", float, dt.focusing_depth),
            power=tr.get("power", float, dt.power),
            amplitude=tr.get("amplitude", float, dt.amplitude),
        )
        f_tr = tr.get("frequency", float, frequency)
        if abs(f_tr - frequency) > 1e-6 * frequency:
            log.warning("Transducer.frequency %.6g Hz differs from BaseFrequency %.6g Hz", f_tr, frequency)

        ab = opts.child("AbsorbingBC")
        width = ab.get("width", float, 0.0)
        power = ab.get("power", float, 2.0)
        kind = ab.get("type", str, "sponge").lower()
        levels = ab.get("levels", int, 4)
        if ab.exists() and ab.get("strength", float) is None and ab.get("reflection", float) is not None and width > 0:
            layer = AbsorbingLayer.for_reflection(width, ab.get("reflection", float, 1e-4), power, kind, levels)
        else:
            layer = AbsorbingLayer(width, ab.get("strength", float, 0.0), power, kind, levels)

        return cls(
            medium,
            frequency,
            lengths,
            (origin[0], origin[1], origin[2]),
            transducer,
            cls._read_source(opts.child("Initial")),
            layer,
        )

    @staticmethod
    def _read_source(ini: Options) -> Source | None:
        if not ini.exists():
            return None
        if ini.get("type", str, "ascii").lower() == "binary":
            files = ini.child("InputFile")
            precision = ini.get("precision", str, "double").lower()
            return Source.from_binary(
                ini.resolve(files.get("xcord", str, "")),
                ini.resolve(files.get("ycord", str, "")),
                [ini.resolve(f) for f in files.get("real", list[str], [])],
                [ini.resolve(f) for f in files.get("imag", list[str], [])],
                np.float32 if precision == "single" else np.float64,
            )
        name = ini.get("InputFile", str)
        if not name:
            return None
        path = ini.resolve(name)
        log.debug("source read from %s", path)
        return Source.from_text(path)

    def make_grid(self, opts: Options | None = None) -> Grid:
        """Build the grid from the ``Solver.DiffractSolver`` options (defaults: 6 points per
        wavelength in x, y and z, 32 samples per period)."""
        opts = opts if opts is not None else Options()
        ppw = opts.get("NumberOfPointsPerWaveLength", list[float], [6.0])
        if len(ppw) not in (1, 3):
            raise TypeError(f"options: '{opts.path}.NumberOfPointsPerWaveLength' should be one or three numbers")
        ppw3 = (ppw[0], ppw[0], ppw[0]) if len(ppw) == 1 else (ppw[0], ppw[1], ppw[2])
        nop = opts.get("NumberOfPoints", list[float], None)
        if nop is not None and len(nop) != 3:
            raise TypeError(f"options: '{opts.path}.NumberOfPoints' should be three numbers (nx, ny, nz)")
        return Grid.from_lengths(
            *self.lengths,
            wavelength=self.wavelength,
            points_per_wavelength=ppw3,
            nt=opts.get("SamplingFrequencyMultiplier", int, 32),
            number_of_points=None if nop is None else (int(nop[0]), int(nop[1]), int(nop[2])),
            origin=self.origin,
            fast_sizes=opts.get("FastFFTSizes", bool, True),
        )

    def initial_field(self, grid: Grid, dtype: type[np.complexfloating] = np.complex128) -> ComplexArray:
        """Harmonics on the source plane, shape (K, nx, ny)."""
        if self.source is not None:
            out = self.source.on_grid(grid, grid.n_harmonics)
        else:
            out = np.zeros((grid.n_harmonics, grid.nx, grid.ny), dtype=complex)
            out[0] = self.transducer.field(grid, self.medium, self.frequency, grid.centre)
        return out.astype(dtype)
