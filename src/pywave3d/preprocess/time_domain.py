"""Output of time-domain pseudo-spectral (k-space) solvers such as k-wave-python.

A time-domain solver records the pressure p(t) at sensor points. For the continuous-wave field
parameters of :class:`~pywave3d.results.Results`, the record has to be reduced to the harmonics
of a steady periodic field on planes:

1. **window**: once the transient has passed the sensors, the field is periodic. The harmonics
   are fitted over the last ``periods`` periods of the record (or from ``window_start``); the
   fit is least squares (:class:`~pywave3d.preprocess._common.HarmonicFit`), exact whatever the
   time step, and its residual is checked: more than 1% of the signal means the window is not
   steady (still transient, or more harmonics than asked for).
2. **harmonics**: P_n for n = 1 .. K, with p = Re sum P_n exp(+i n omega t). K must stay below
   the Nyquist frequency of the record, and is in practice limited by the points per wavelength
   of the grid (k-Wave resolves up to about ppw / 2 harmonics).
3. **planes**: the axis along which the beam travels (``propagation``) becomes z, the other
   one or two axes x and y, in their order; the phase is made retarded, exp(+i n k (z - z0)).

k-Wave specifics (:func:`from_kwave`): ``kspaceFirstOrder`` returns ``sensor_data["p"]`` as
(n_sensor, Nt) with the sensor points in C order of the mask, which must be a rectangular block
(or ``None``, the whole grid). Recording starts at ``sensor.record_start_index`` (1-based), and
the coordinates are ``kgrid.x_vec`` etc. With ``pml_inside=False`` (k-Wave's default) the
recorded grid excludes the PML. The pressure is collocated on the grid points (only the
velocity is staggered), so no shift is needed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from pywave3d._types import RealArray
from pywave3d.preprocess._common import record, time_series_planes
from pywave3d.recorder import RecordSettings
from pywave3d.results import Results


def from_time_series(
    p: Any,
    dt: float,
    *,
    coords: Mapping[str, Any],
    layout: str = "xyzt",
    propagation: str = "z",
    frequency: float,
    sound_speed: float,
    density: float,
    n_harmonics: int = 4,
    periods: float | None = 2.0,
    window_start: float | None = None,
    start_time: float = 0.0,
    detrend: bool = False,
    settings: RecordSettings | None = None,
    axis_xy: tuple[float, float] | None = None,
    alpha: RealArray | None = None,
    focusing_depth: float = 0.0,
) -> Results:
    """:class:`Results` of pressure time series on a regular grid.

    Args:
        p: pressure [Pa] with axes named by ``layout``: two or three of ``x``, ``y``, ``z`` and
            the time axis ``t`` (e.g. ``"xyzt"``, ``"txz"``).
        dt: time step of the record [s].
        coords: coordinate vector of each space axis [m], keyed by its letter; uniformly
            spaced and increasing.
        layout: names of the axes of ``p``.
        propagation: the axis along which the beam travels; it becomes z.
        frequency: base frequency f0 [Hz].
        sound_speed, density: c0 [m/s] and rho0 [kg/m^3], for intensities and the retarded time.
        n_harmonics: K, harmonics fitted.
        periods: length of the fitting window, in periods, ending at the end of the record.
        window_start: alternatively, the time from which to fit [s] (same clock as
            ``start_time``).
        start_time: time of the first sample [s].
        detrend: also fit a linear drift (see :mod:`pywave3d.preprocess.fdtd`).
        settings, axis_xy, alpha, focusing_depth: as in
            :func:`pywave3d.preprocess.spectral.from_harmonics`.
    """
    grid, planes = time_series_planes(
        p,
        layout,
        coords,
        dt,
        frequency=frequency,
        sound_speed=sound_speed,
        n_harmonics=n_harmonics,
        propagation=propagation,
        periods=periods,
        start_time=start_time,
        window_start=window_start,
        detrend=detrend,
    )
    return record(
        planes,
        grid,
        frequency=frequency,
        sound_speed=sound_speed,
        density=density,
        settings=settings,
        axis_xy=axis_xy,
        alpha=alpha,
        focusing_depth=focusing_depth,
    )


def sensor_block(mask: Any, shape: tuple[int, ...]) -> tuple[slice, ...]:
    """The slices of the rectangular block covered by a binary sensor mask of ``shape``."""
    m = np.asarray(mask).astype(bool)
    if m.size == 1:  # a scalar True: the whole grid
        m = np.broadcast_to(m.reshape(()), shape)
    m = m.reshape(shape)
    idx = np.nonzero(m)
    if idx[0].size == 0:
        raise ValueError("the sensor mask is empty")
    block = tuple(slice(int(i.min()), int(i.max()) + 1) for i in idx)
    if not m[block].all() or np.count_nonzero(m) != m[block].size:
        raise ValueError("the sensor mask must be a rectangular block of grid points (a plane or volume)")
    return block


def kwave_coordinates(kgrid: Any) -> dict[str, RealArray]:
    """Coordinate vectors of a ``kWaveGrid``, keyed ``x``, ``y``, ``z`` (as many as its dimensions)."""
    dim = int(kgrid.dim)
    return {c: np.asarray(getattr(kgrid, f"{c}_vec"), dtype=float).ravel() for c in "xyz"[:dim]}


def from_kwave(
    sensor_data: Mapping[str, Any] | Any,
    kgrid: Any,
    *,
    frequency: float,
    sound_speed: float,
    density: float,
    sensor_mask: Any = None,
    record_start_index: int = 1,
    propagation: str = "x",
    n_harmonics: int = 4,
    periods: float | None = 2.0,
    window_start: float | None = None,
    order: str = "C",
    settings: RecordSettings | None = None,
    axis_xy: tuple[float, float] | None = None,
    alpha: RealArray | None = None,
    focusing_depth: float = 0.0,
) -> Results:
    """:class:`Results` of a k-wave-python simulation of a continuous-wave source.

    Args:
        sensor_data: what ``kspaceFirstOrder`` returned (a dict with ``"p"``), or the array
            ``p`` itself, shaped (n_sensor, Nt_recorded).
        kgrid: the ``kWaveGrid`` (``dim``, ``x_vec`` .., ``dt``).
        frequency: the source frequency [Hz].
        sound_speed, density: c0 and rho0 of the medium (for intensities and retarded time).
        sensor_mask: the binary ``sensor.mask`` (a rectangular block), or None for the whole
            grid.
        record_start_index: ``sensor.record_start_index`` (1-based, as in k-Wave).
        propagation: the grid axis along which the beam travels (k-Wave examples often use x).
        n_harmonics, periods, window_start, settings, axis_xy, alpha, focusing_depth: as in
            :func:`from_time_series`.
        order: ``"C"`` for the sensor ordering of ``kspaceFirstOrder`` (k-wave-python >= 0.6);
            ``"F"`` for MATLAB's column-major order (results read from MATLAB files).
    """
    p = sensor_data["p"] if isinstance(sensor_data, Mapping) else sensor_data
    p = np.asarray(p, dtype=float)
    coords = kwave_coordinates(kgrid)
    shape = tuple(v.size for v in coords.values())
    if sensor_mask is None:
        block = tuple(slice(0, n) for n in shape)
    else:
        block = sensor_block(np.asarray(sensor_mask), shape)
    sub = tuple(s.stop - s.start for s in block)
    if p.ndim != 2 or p.shape[0] != int(np.prod(sub)):
        raise ValueError(f"sensor data of shape {p.shape}: expected ({int(np.prod(sub))}, Nt) for the block {sub}")
    series = p.reshape((*sub, p.shape[1]), order=order)  # type: ignore[call-overload]
    coords = {c: v[s] for (c, v), s in zip(coords.items(), block, strict=True)}
    dt = float(np.asarray(kgrid.dt).ravel()[0])
    layout = "".join(coords) + "t"
    return from_time_series(
        series,
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
        start_time=(record_start_index - 1) * dt,
        settings=settings,
        axis_xy=axis_xy,
        alpha=alpha,
        focusing_depth=focusing_depth,
    )
