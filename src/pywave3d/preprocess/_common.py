"""What the preprocessors share: layouts, grids, harmonic analysis and feeding the recorder."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from pywave3d._types import ComplexArray, RealArray
from pywave3d.grid import Grid
from pywave3d.recorder import PlaneRecorder, RecordSettings
from pywave3d.results import Results

log = logging.getLogger("pywave3d.preprocess")

#: residual of the harmonic fit, relative to the signal, above which the record is not steady
STEADY_TOLERANCE = 1e-2


def parse_layout(layout: str, ndim: int, allowed: str) -> str:
    """Check that ``layout`` names each axis of an ``ndim`` array once, from ``allowed``."""
    if len(layout) != ndim or len(set(layout)) != ndim or not set(layout) <= set(allowed):
        raise ValueError(f"layout {layout!r} does not describe a {ndim}-dimensional array with axes from {allowed!r}")
    return layout


def coordinate_vector(values: Any, name: str) -> RealArray:
    """A 1D float array of coordinates, which must be uniformly spaced and increasing."""
    v = np.asarray(values, dtype=float).ravel()
    if v.size > 1:
        d = np.diff(v)
        if np.any(d <= 0) or not np.allclose(d, d[0], rtol=1e-6, atol=0.0):
            raise ValueError(f"the {name} coordinates must be uniformly spaced and increasing")
    return v


def plane_grid(x: RealArray, y: RealArray | None, z: RealArray, n_harmonics: int) -> Grid:
    """The :class:`Grid` of planes at ``z`` over (x, y); ``y`` None for a 2D field (ny = 1, dy = 1 m)."""
    if x.size < 2:
        raise ValueError("a plane needs at least 2 points across")
    dz = float(np.mean(np.diff(z))) if z.size > 1 else 1.0
    if y is None:
        ny, dy, y0 = 1, 1.0, 0.0
    else:
        ny, dy, y0 = y.size, float(y[1] - y[0]) if y.size > 1 else 1.0, float(y[0])
    return Grid(
        nx=x.size,
        ny=ny,
        nz=z.size,
        nt=2 * n_harmonics,
        dx=float(x[1] - x[0]),
        dy=dy,
        dz=dz,
        x0=float(x[0]),
        y0=y0,
        z0=float(z[0]),
    )


def axis_index(grid: Grid, axis_xy: tuple[float, float] | None) -> tuple[int, int] | None:
    """Indices of the beam axis nearest ``axis_xy`` (None: the plane centre)."""
    if axis_xy is None:
        return None
    return grid.nearest(axis_xy[0], axis_xy[1] if grid.ny > 1 else grid.y0)  # type: ignore[return-value]


def to_retarded(harmonics: ComplexArray, z: float, z0: float, frequency: float, sound_speed: float) -> ComplexArray:
    """Harmonics (K, ...) in absolute phase on the plane z, in the retarded time tau = t - (z - z0) / c0.

    With p = Re sum P_n exp(+i n omega t), shifting to tau multiplies P_n by exp(+i n k (z - z0)).
    """
    n = np.arange(1, harmonics.shape[0] + 1).reshape((-1,) + (1,) * (harmonics.ndim - 1))
    return harmonics * np.exp(1j * n * (2 * np.pi * frequency / sound_speed) * (z - z0))


def record(
    planes: Iterable[tuple[float, ComplexArray]],
    grid: Grid,
    *,
    frequency: float,
    sound_speed: float,
    density: float,
    settings: RecordSettings | None,
    axis_xy: tuple[float, float] | None,
    alpha: RealArray | None,
    focusing_depth: float = 0.0,
) -> Results:
    """Feed ``(z, harmonics)`` planes, in increasing z, to a :class:`PlaneRecorder`."""
    recorder = PlaneRecorder(
        grid,
        grid.n_harmonics,
        frequency=frequency,
        sound_speed=sound_speed,
        impedance=density * sound_speed,
        settings=settings,
        axis_index=axis_index(grid, axis_xy),
    )
    for z, h in planes:
        recorder.add(z, h)
    extra: dict[str, Any] = {"focusing_depth": focusing_depth}
    if alpha is not None:
        extra["alpha"] = np.asarray(alpha, dtype=float)
    return recorder.finish(**extra)


class HarmonicFit:
    """Least-squares fit of the harmonics of f0 to sampled time series.

    The window is the last ``periods`` periods of the record (or the samples from
    ``start_time``), and the model is

        p(t) = c0 [+ c1 (t - t_mid)] + sum_n Re(P_n exp(+i n omega t)),  n = 1 .. K,

    with the optional linear term (``detrend``) absorbing slow drift. Unlike an FFT over the
    window, the fit is exact for any time step, whether or not a period holds a whole number of
    samples. The fit is linear, so its pseudo-inverse is computed once and applied to every
    series.

    Args:
        n_samples: samples in each series.
        dt: time step [s].
        frequency: f0 [Hz].
        n_harmonics: K.
        periods: length of the window in periods (from the end of the record).
        start_time: time of the first sample [s] (sets the phase origin, t = 0).
        window_start: alternatively, the time from which the window runs to the end.
        detrend: also fit a linear drift.
    """

    def __init__(
        self,
        n_samples: int,
        dt: float,
        frequency: float,
        n_harmonics: int,
        *,
        periods: float | None = 2.0,
        start_time: float = 0.0,
        window_start: float | None = None,
        detrend: bool = False,
    ) -> None:
        if n_harmonics * frequency >= 0.5 / dt:
            raise ValueError(
                f"harmonic {n_harmonics} ({n_harmonics * frequency:.4g} Hz) is above the Nyquist frequency "
                f"{0.5 / dt:.4g} Hz of the record: ask for fewer harmonics"
            )
        t = start_time + dt * np.arange(n_samples)
        if window_start is not None:
            first = int(np.searchsorted(t, window_start - 1e-9 * dt))
        else:
            assert periods is not None
            first = n_samples - round(periods / (frequency * dt))
        if first < 0 or n_samples - first < 2 * n_harmonics + 2:
            raise ValueError("the record is too short for the window and harmonics asked for")
        self.first = first
        tw = t[first:]
        w = 2 * np.pi * frequency
        cols = [np.ones_like(tw)]
        if detrend:
            cols.append(tw - tw.mean())
        self._offset = len(cols)
        for n in range(1, n_harmonics + 1):
            cols += [np.cos(n * w * tw), np.sin(n * w * tw)]
        self._design = np.column_stack(cols)
        self._pinv = np.linalg.pinv(self._design)
        self.n_harmonics = n_harmonics
        self.window_periods = float((tw[-1] - tw[0] + dt) * frequency)
        if self.window_periods < 1 - 1e-9:
            log.warning("the harmonic fit window holds %.2f periods: less than one", self.window_periods)

    def __call__(self, series: RealArray, time_axis: int = -1) -> tuple[ComplexArray, float]:
        """Return the harmonics (K, ...) of ``series`` and the relative rms residual of the fit."""
        s = np.moveaxis(np.asarray(series, dtype=float), time_axis, 0)
        rest = s.shape[1:]
        win = s[self.first :].reshape(s.shape[0] - self.first, -1)
        coef = self._pinv @ win
        a = coef[self._offset :: 2]
        b = coef[self._offset + 1 :: 2]
        harmonics = (a - 1j * b).reshape((self.n_harmonics, *rest))  # Re(P e^{iwt}) = a cos - b' sin
        resid = win - self._design @ coef
        norm = float(np.sqrt(np.mean(win**2)))
        rel = float(np.sqrt(np.mean(resid**2)) / norm) if norm > 0 else 0.0
        return harmonics, rel


def time_series_planes(
    p: Any,
    layout: str,
    coords: Mapping[str, Any],
    dt: float,
    *,
    frequency: float,
    sound_speed: float,
    n_harmonics: int,
    propagation: str,
    periods: float | None,
    start_time: float,
    window_start: float | None,
    detrend: bool,
) -> tuple[Grid, Iterable[tuple[float, ComplexArray]]]:
    """Grid and (z, harmonics) planes of gridded time series ``p`` (axes named by ``layout``)."""
    arr = np.asarray(p)
    parse_layout(layout, arr.ndim, "xyzt")
    if "t" not in layout:
        raise ValueError("the layout must name the time axis 't'")
    space = [c for c in layout if c != "t"]
    if propagation not in space:
        raise ValueError(f"the propagation axis {propagation!r} is not one of the axes {space}")
    transverse = [c for c in space if c != propagation]
    if not 1 <= len(transverse) <= 2:
        raise ValueError("the field needs one or two transverse axes")
    order = [layout.index(propagation)] + [layout.index(c) for c in transverse] + [layout.index("t")]
    arr = np.transpose(arr, order)  # (z, x[, y], t)
    vec = {}
    for c in space:
        if c not in coords:
            raise ValueError(f"no coordinates for axis {c!r}")
        vec[c] = coordinate_vector(coords[c], c)
        length = arr.shape[[propagation, *transverse].index(c)]
        if vec[c].size != length:
            raise ValueError(f"{vec[c].size} coordinates for axis {c!r} of length {length}")
    z = vec[propagation]
    x = vec[transverse[0]]
    y = vec[transverse[1]] if len(transverse) == 2 else None
    grid = plane_grid(x, y, z, n_harmonics)
    fit = HarmonicFit(
        arr.shape[-1],
        dt,
        frequency,
        n_harmonics,
        periods=periods,
        start_time=start_time,
        window_start=window_start,
        detrend=detrend,
    )
    log.info(
        "harmonic fit over %.2f periods (%d samples), %d harmonics",
        fit.window_periods,
        arr.shape[-1] - fit.first,
        n_harmonics,
    )

    def planes() -> Iterable[tuple[float, ComplexArray]]:
        worst = 0.0
        for iz, zi in enumerate(z):
            series = arr[iz] if y is not None else arr[iz][:, None, :]  # (nx, ny, nt)
            h, rel = fit(series, time_axis=-1)
            worst = max(worst, rel)
            yield float(zi), to_retarded(h, float(zi), float(z[0]), frequency, sound_speed)
        if worst > STEADY_TOLERANCE:
            log.warning(
                "the harmonic fit leaves %.2g of the signal (rms) on some plane: the record is not a "
                "steady periodic field over the window (transient, too few harmonics, or noise)",
                worst,
            )

    return grid, planes()
