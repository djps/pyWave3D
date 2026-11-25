"""Output of frequency-domain pseudo-spectral solvers: harmonics on planes.

Spatial-marching codes (pyWave3D, the MSU HIFU beam codes, KZK and
angular-spectrum solvers such as k-Wave's ``angularSpectrumCW``) produce what
:class:`~pywave3d.results.Results` needs directly: the complex amplitude of each harmonic of a
periodic field on transverse planes. Preprocessing only has to

* put the array in the order (z, harmonic, x, y), from any order named by ``layout``;
* make the phase convention p = Re sum P_n exp(+i n omega tau), retarded or not: a solver using
  exp(-i omega t) gives the complex conjugate (``time_sign=-1``), and absolute phases
  (``phase="absolute"``, P ~ exp(-i n k z) for a wave travelling along +z) are turned into
  retarded ones by exp(+i n k (z - z0));
* describe the grid: uniformly spaced x (and y), planes in increasing z.

A single-frequency (CW) field has no harmonic axis: leave ``k`` out of the layout. A 2D field
has no y axis: leave ``y`` out; its plane quantities are then per metre in y.

Example, a k-Wave ``angular_spectrum_cw`` result of shape (nx, ny, nz)::

    res = from_harmonics(p, layout="xyz", x=x, y=y, z=z, frequency=1e6, sound_speed=1500,
                         density=1000, phase="absolute")
    res.metrics(), res.save("out")
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

import numpy as np

from pywave3d._types import ComplexArray, RealArray
from pywave3d.preprocess._common import coordinate_vector, parse_layout, plane_grid, record, to_retarded
from pywave3d.recorder import RecordSettings
from pywave3d.results import Results

Phase = Literal["retarded", "absolute"]


def standard_layout(harmonics: Any, layout: str) -> ComplexArray:
    """Return ``harmonics`` as an array (nz, K, nx, ny), from axes named by ``layout``.

    ``layout`` names each axis with one of ``z`` (planes), ``k`` (harmonics), ``x``, ``y``;
    ``k`` and ``y`` may be absent (one harmonic, a 2D field).
    """
    arr = np.asarray(harmonics)
    parse_layout(layout, arr.ndim, "zkxy")
    if "z" not in layout or "x" not in layout:
        raise ValueError("the layout must name the plane axis 'z' and the transverse axis 'x'")
    full = layout + "".join(c for c in "ky" if c not in layout)
    arr = arr.reshape(arr.shape + (1,) * (len(full) - arr.ndim))
    return np.transpose(arr, [full.index(c) for c in "zkxy"]).astype(complex, copy=False)


def from_harmonics(
    harmonics: Any,
    *,
    x: Any,
    z: Any,
    y: Any = None,
    layout: str = "zkxy",
    frequency: float,
    sound_speed: float,
    density: float,
    phase: Phase = "retarded",
    time_sign: int = 1,
    settings: RecordSettings | None = None,
    axis_xy: tuple[float, float] | None = None,
    alpha: RealArray | None = None,
    focusing_depth: float = 0.0,
) -> Results:
    """:class:`Results` of a field given by its harmonics on planes.

    Args:
        harmonics: complex amplitudes [Pa], axes named by ``layout``.
        x, y, z: coordinates of the axes [m] (``y`` None for a 2D field); x and y uniformly
            spaced, z increasing (it may be non-uniform).
        layout: order of the axes of ``harmonics`` (see :func:`standard_layout`).
        frequency: base frequency f0 [Hz]; harmonic n is at n f0.
        sound_speed, density: c0 [m/s] and rho0 [kg/m^3] of the medium.
        phase: ``"retarded"`` (tau = t - z / c0, as pyWave3D) or ``"absolute"``.
        time_sign: +1 if the solver's field is p = Re P exp(+i omega t), -1 for exp(-i omega t).
        settings: what to record (points, focal quantity, levels, ...).
        axis_xy: the beam axis (x, y) [m]; default the plane centre.
        alpha: absorption of each harmonic [Np/m], for the heating rate.
        focusing_depth: geometric focus [m], for the focal shift (0 if unknown).
    """
    if time_sign not in (1, -1):
        raise ValueError("time_sign must be +1 or -1")
    if phase not in ("retarded", "absolute"):
        raise ValueError("phase must be 'retarded' or 'absolute'")
    h = standard_layout(harmonics, layout)
    if time_sign == -1:
        h = h.conj()
    xv = coordinate_vector(x, "x")
    yv = coordinate_vector(y, "y") if y is not None else None
    zv = np.asarray(z, dtype=float).ravel()
    if np.any(np.diff(zv) <= 0):
        raise ValueError("the z coordinates must increase")
    nz, k, nx, ny = h.shape
    if (zv.size, xv.size, 1 if yv is None else yv.size) != (nz, nx, ny):
        raise ValueError(f"coordinates ({zv.size}, {xv.size}, {None if yv is None else yv.size}) do not fit {h.shape}")
    grid = plane_grid(xv, yv, zv, k)

    def planes() -> Iterable[tuple[float, ComplexArray]]:
        for iz in range(nz):
            p = h[iz]
            if phase == "absolute":
                p = to_retarded(p, float(zv[iz]), float(zv[0]), frequency, sound_speed)
            yield float(zv[iz]), p

    return record(
        planes(),
        grid,
        frequency=frequency,
        sound_speed=sound_speed,
        density=density,
        settings=settings,
        axis_xy=axis_xy,
        alpha=alpha,
        focusing_depth=focusing_depth,
    )
