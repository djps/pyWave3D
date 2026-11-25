"""Output of finite-difference time-domain (FDTD) solvers such as fullwave25.

The reduction is that of :mod:`pywave3d.preprocess.time_domain` (a steady window of the record,
least-squares harmonics, planes along the propagation axis, retarded phase), with what differs
for FDTD codes:

* **drift**: absorbing layers and relaxation-based attenuation leave a slow, low-frequency drift
  in the recorded pressure (fullwave25 offers a high-pass filter against it). The harmonic fit
  includes a constant and a linear term by default (``detrend=True``), which absorbs it over the
  window without filtering the harmonics.
* **time decimation**: the record is often kept every ``sampling_modulus_time`` steps; the
  time step of the record is then ``dt * sampling_modulus_time``, and its Nyquist frequency
  limits the harmonics that can be fitted.
* **staggered grids**: FDTD codes keep pressure and particle velocity on staggered grids. The
  pressure sits on the integer nodes (fullwave25 records it there), so its coordinates are the
  node indices times the spacing; a code recording pressure at half nodes needs the
  coordinates shifted by half a cell, through ``coords``.
* **numerical dispersion**: the phase speed of an FDTD scheme is slightly below c0 and depends
  on direction. Magnitudes, peaks and intensities are unaffected; the retarded phase of the
  planes drifts slowly along z, which only shifts the output waveforms in time.
* **sensors as indices**: fullwave25 records the sensors listed in ``sensor.outcoords``
  (grid indices, in ``np.argwhere`` order of the mask) on the domain without its absorbing
  layer, with x the axial (depth) direction, y lateral and z elevational. The data are put on
  the grid by those indices, so their order does not matter, but they must fill a rectangular
  block. Its "sparse-grid" whole-domain recording orders the points inside the compiled solver
  and is not supported: record with a mask instead.

fullwave25 itself is not needed: :func:`from_fullwave` reads only the attributes named below.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from pywave3d._types import IntArray, RealArray
from pywave3d.preprocess.time_domain import from_time_series as _from_time_series
from pywave3d.recorder import RecordSettings
from pywave3d.results import Results


def from_time_series(
    p: Any,
    dt: float,
    *,
    coords: Mapping[str, Any],
    layout: str = "xyt",
    propagation: str = "x",
    frequency: float,
    sound_speed: float,
    density: float,
    n_harmonics: int = 4,
    periods: float | None = 2.0,
    window_start: float | None = None,
    start_time: float = 0.0,
    detrend: bool = True,
    settings: RecordSettings | None = None,
    axis_xy: tuple[float, float] | None = None,
    alpha: RealArray | None = None,
    focusing_depth: float = 0.0,
) -> Results:
    """:class:`Results` of FDTD pressure time series on a regular grid of nodes.

    As :func:`pywave3d.preprocess.time_domain.from_time_series`, with the FDTD defaults: a 2D
    record ``(x, y, t)`` with x the axial direction, and the drift removed (``detrend``).
    """
    return _from_time_series(
        p,
        dt,
        coords=coords,
        layout=layout,
        propagation=propagation,
        frequency=frequency,
        sound_speed=sound_speed,
        density=density,
        n_harmonics=n_harmonics,
        periods=periods,
        window_start=window_start,
        start_time=start_time,
        detrend=detrend,
        settings=settings,
        axis_xy=axis_xy,
        alpha=alpha,
        focusing_depth=focusing_depth,
    )


def gather(sensor_data: Any, indices: Any) -> tuple[RealArray, IntArray]:
    """Put sensor series (n_sensors, nt) on the block of grid indices (n_sensors, ndim) they cover.

    Returns the block (shape (*block, nt)) and its first index along each axis. Raises
    ValueError if the indices do not fill a rectangular block exactly once.
    """
    data = np.asarray(sensor_data, dtype=float)
    idx = np.asarray(indices, dtype=np.int64)
    if idx.ndim != 2 or data.ndim != 2 or data.shape[0] != idx.shape[0]:
        raise ValueError(f"{data.shape[0]} series for {idx.shape[0]} sensor positions")
    lo = idx.min(axis=0)
    shape = tuple(int(n) for n in idx.max(axis=0) - lo + 1)
    if int(np.prod(shape)) != idx.shape[0]:
        raise ValueError(f"the {idx.shape[0]} sensors do not fill the {shape} block they span: record a block")
    out = np.full((*shape, data.shape[1]), np.nan)
    rel = tuple((idx - lo).T)
    out[rel] = data
    if np.isnan(out).any():
        raise ValueError("some sensor positions are repeated: the block is not filled")
    return out, lo


def from_fullwave(
    sensor_data: Any,
    sensor: Any,
    grid: Any,
    *,
    density: float,
    frequency: float | None = None,
    sound_speed: float | None = None,
    propagation: str = "x",
    origin: tuple[float, ...] | None = None,
    n_harmonics: int = 4,
    periods: float | None = 2.0,
    window_start: float | None = None,
    detrend: bool = True,
    settings: RecordSettings | None = None,
    axis_xy: tuple[float, float] | None = None,
    alpha: RealArray | None = None,
    focusing_depth: float = 0.0,
) -> Results:
    """:class:`Results` of a fullwave25 simulation of a continuous-wave source.

    Args:
        sensor_data: what ``fullwave.Solver.run()`` returned: (n_sensors, nt_recorded).
        sensor: the ``fullwave.Sensor`` (``outcoords``, ``sampling_modulus_time``,
            ``is_sparse_grid``).
        grid: the ``fullwave.Grid`` (``dx``, ``dy``, ``dz``, ``dt``, ``f0``, ``c0``).
        density: rho0 [kg/m^3] where the beam is measured.
        frequency: source frequency [Hz] (default ``grid.f0``).
        sound_speed: c0 [m/s] where the beam is measured (default ``grid.c0``, which fullwave25
            uses only to size the grid: pass the medium's value if it differs).
        propagation: the axis along which the beam travels (fullwave25's axial axis is x).
        origin: coordinates of grid index 0 along each axis [m] (default 0).
        n_harmonics, periods, window_start, detrend, settings, axis_xy, alpha, focusing_depth:
            as in :func:`from_time_series`.
    """
    if getattr(sensor, "is_sparse_grid", False):
        raise NotImplementedError("fullwave25 sparse-grid (whole-domain) records are not supported: use a sensor mask")
    f0 = float(grid.f0) if frequency is None else frequency
    c0 = float(grid.c0) if sound_speed is None else sound_speed
    block, lo = gather(sensor_data, sensor.outcoords)
    ndim = block.ndim - 1
    names = "xyz"[:ndim]
    spacing = [float(getattr(grid, f"d{c}")) for c in names]
    o = origin if origin is not None else (0.0,) * ndim
    coords = {c: o[a] + spacing[a] * (lo[a] + np.arange(block.shape[a])) for a, c in enumerate(names)}
    dt = float(grid.dt) * int(getattr(sensor, "sampling_modulus_time", 1))
    return from_time_series(
        block,
        dt,
        coords=coords,
        layout=names + "t",
        propagation=propagation,
        frequency=f0,
        sound_speed=c0,
        density=density,
        n_harmonics=n_harmonics,
        periods=periods,
        window_start=window_start,
        detrend=detrend,
        settings=settings,
        axis_xy=axis_xy,
        alpha=alpha,
        focusing_depth=focusing_depth,
    )
