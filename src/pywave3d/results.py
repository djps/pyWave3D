"""What a run records, the field metrics derived from it, and writing them to files.

The recorded data are those needed for the field parameters of the IEC standards (see
``docs/iec_outputs.md``): the beam axis, the two lines through it (x and y), extreme pressures
over every plane, the power through every plane, and two whole planes: the plane of the focus
(maximum of the focal quantity on the beam axis) and the plane of the spatial peak of p+.

:class:`Results` does not depend on the solver. It is filled by
:class:`pywave3d.recorder.PlaneRecorder` from planes of harmonics, which the marching solver
produces directly and :mod:`pywave3d.preprocess` makes from the output of other solvers
(frequency-domain pseudo-spectral, time-domain pseudo-spectral such as k-wave-python, FDTD such as
fullwave25). A 2D field has ``grid.ny == 1``: its y widths are NaN and, with ``grid.dy = 1``, its
power and areas are per metre in y.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from pywave3d import iec
from pywave3d._types import ComplexArray, RealArray
from pywave3d.grid import Grid
from pywave3d.nonlinear import peak_pressures, waveform

FocalQuantity = Literal["p_positive", "p_negative", "first_harmonic"]


def width_at_level(coord: RealArray, values: RealArray, index: int, level: float) -> float:
    """Width of the region around ``index`` where ``values >= level``, interpolated linearly.

    Returns NaN if the region reaches an end of the array (the width is not resolved).
    """
    v = np.asarray(values, dtype=float)
    lo = hi = index
    while lo > 0 and v[lo - 1] >= level:
        lo -= 1
    while hi < v.size - 1 and v[hi + 1] >= level:
        hi += 1
    if lo == 0 or hi == v.size - 1:
        return math.nan
    cl = coord[lo - 1] + (level - v[lo - 1]) / (v[lo] - v[lo - 1]) * (coord[lo] - coord[lo - 1])
    cu = coord[hi] + (level - v[hi]) / (v[hi + 1] - v[hi]) * (coord[hi + 1] - coord[hi])
    return float(cu - cl)


def connected_area(values: RealArray, seed: tuple[int, int], level: float, cell: float) -> float:
    """Area [m^2] of the 4-connected region around ``seed`` where ``values >= level``."""
    from scipy.ndimage import label

    labels, _ = label(values >= level)
    lab = labels[seed]
    return float(np.count_nonzero(labels == lab) * cell) if lab else 0.0


@dataclass
class Results:
    """Quantities recorded on the planes of a run.

    Pressures are in Pa, lengths in m, intensities in W/m^2. Peak pressures are those of
    the band-limited waveform (the K harmonics kept), sampled at 4 points per period per active harmonic.
    Intensities use the plane-wave relation I = sum_n |P_n|^2 / (2 rho0 c0), as hydrophone
    measurements do (IEC 62127-1).

    Attributes:
        grid: the grid of the run.
        z: recorded planes, shape (nrec,).
        axis: harmonics on the beam axis (``axis_ij``), shape (nrec, K).
        axis_p_plus, axis_p_minus: peak positive / negative pressure on the axis, (nrec,).
        xz_p_plus, xz_p_minus: the same along x through the axis, (nrec, nx).
        yz_p_plus, yz_p_minus: the same along y through the axis, (nrec, ny).
        xz_amplitude, yz_amplitude: |P_1| along x / y through the axis.
        plane_p_plus, plane_p_minus: extreme peak pressures over each plane, (nrec,) (only
            with ``plane_peaks``; else NaN).
        power: acoustic power through each plane [W], (nrec,).
        plane_intensity: maximum intensity over each plane, (nrec,).
        max_p_plus: maximum over z of the peak positive pressure, (nx, ny) (``plane_peaks``).
        final: harmonics on the last plane, (K, nx, ny).
        focal_index: index in ``z`` of the focus: the maximum of ``focal_quantity`` on the axis.
        focal_plane: harmonics on the focal plane, (K, nx, ny).
        peak_index, peak_xy_index: plane and (i, j) of the spatial peak of p+ (-1 and the axis
            without ``plane_peaks``).
        peak_plane: harmonics on the plane of the spatial peak (None without ``plane_peaks``).
        points: harmonics at the requested points, by output name.
        source_amplitude: max |P_1| on the source plane.
        focusing_depth: geometric focus of the transducer (0 if unknown).
        alpha: absorption of each harmonic [Np/m], (K,) (for the heating rate).
        centroid_x, centroid_y: intensity centroid of the -6 dB region of each plane, (nrec,).
        source_plane: harmonics on the source plane.
        face_index, face_plane: the plane nearest ``face_distance`` from the source (IEC 61689).
        sound_speed: c0 [m/s].
        levels_db: levels of the widths, lengths and areas (the first is the main one).
        metrics_harmonics: harmonics reported in the metrics and files.
        hydrophone_diameter: diameter of the simulated hydrophone [m] (0: none).
        derating: derating coefficient [dB / (cm MHz)].
        godunov_from: first z at which the Godunov scheme was used (NaN if never).
        timings: seconds spent per part of the step.
        axis_index: (i, j) of the beam axis on the plane (None: the plane centre).
    """

    grid: Grid
    z: RealArray
    axis: ComplexArray
    axis_p_plus: RealArray
    axis_p_minus: RealArray
    xz_p_plus: RealArray
    xz_p_minus: RealArray
    yz_p_plus: RealArray
    yz_p_minus: RealArray
    xz_amplitude: RealArray
    yz_amplitude: RealArray
    plane_p_plus: RealArray
    plane_p_minus: RealArray
    power: RealArray
    plane_intensity: RealArray
    max_p_plus: RealArray | None
    final: ComplexArray
    focal_index: int = 0
    focal_plane: ComplexArray | None = None
    peak_index: int = -1
    peak_xy_index: tuple[int, int] = (0, 0)
    peak_plane: ComplexArray | None = None
    focal_quantity: FocalQuantity = "p_positive"
    impedance: float = 1.479e6
    points: dict[str, tuple[tuple[float, float, float], ComplexArray]] = field(default_factory=dict)
    source_amplitude: float = 0.0
    focusing_depth: float = 0.0
    frequency: float = 1.0e6
    alpha: RealArray | None = None
    centroid_x: RealArray | None = None
    centroid_y: RealArray | None = None
    source_plane: ComplexArray | None = None
    face_index: int = 0
    face_plane: ComplexArray | None = None
    sound_speed: float = 1482.0
    levels_db: list[float] = field(default_factory=lambda: [-6.0])
    metrics_harmonics: int = 4
    hydrophone_diameter: float = 0.0
    derating: float = iec.DERATING_DB_PER_CM_MHZ
    godunov_from: float = math.nan
    timings: dict[str, float] = field(default_factory=dict)
    axis_index: tuple[int, int] | None = None

    @property
    def axis_ij(self) -> tuple[int, int]:
        """(i, j) of the beam axis on the plane."""
        return self.axis_index if self.axis_index is not None else self.grid.centre_index

    @property
    def axis_xy(self) -> tuple[float, float]:
        """Coordinates (x, y) of the beam axis [m]."""
        i, j = self.axis_ij
        return float(self.grid.x[i]), float(self.grid.y[j])

    # -- profiles ------------------------------------------------------------------------

    def axial(self, quantity: FocalQuantity | None = None) -> RealArray:
        """Return the axial profile of ``quantity`` (magnitudes, Pa); default: the focal quantity."""
        quantity = quantity or self.focal_quantity
        if quantity == "p_positive":
            return self.axis_p_plus
        if quantity == "p_negative":
            return np.abs(self.axis_p_minus)
        if quantity == "first_harmonic":
            return np.abs(self.axis[:, 0])
        raise ValueError(f"unknown focal quantity {quantity!r}")

    def intensity(self, harmonics: ComplexArray, axis: int = 0) -> RealArray:
        """Plane-wave intensity sum_n |P_n|^2 / (2 rho0 c0) of harmonics along ``axis``."""
        return (np.abs(harmonics) ** 2).sum(axis=axis) / (2 * self.impedance)

    @property
    def axial_intensity(self) -> RealArray:
        """Intensity on the beam axis [W/m^2], (nrec,)."""
        return self.intensity(self.axis, axis=1)

    @property
    def axial_heating(self) -> RealArray:
        """Heat deposition rate q = sum_n 2 alpha_n I_n on the axis [W/m^3], (nrec,)."""
        if self.alpha is None:
            return np.full(self.z.size, np.nan)
        return (2 * self.alpha[None, :] * np.abs(self.axis) ** 2).sum(axis=1) / (2 * self.impedance)

    def plane_maps(self, harmonics: ComplexArray) -> dict[str, RealArray]:
        """p+, p-, |P_n| and intensity over a plane of harmonics (K, nx, ny)."""
        p_positive, p_negative = peak_pressures(harmonics)
        return {
            "p_positive": p_positive,
            "p_negative": p_negative,
            "amplitude": np.abs(harmonics),
            "intensity": self.intensity(harmonics),
        }

    def waveform_at(self, harmonics: ComplexArray, nt: int | None = None) -> tuple[RealArray, RealArray]:
        """Return (t, p(t)) over one period from the harmonics of a point."""
        nt = nt or 8 * harmonics.shape[0]
        t = np.arange(nt) / (nt * self.frequency)
        return t, waveform(harmonics.reshape(-1, 1), nt)[:, 0]

    # -- metrics -------------------------------------------------------------------------

    @property
    def density(self) -> float:
        """rho0 [kg/m^3]."""
        return self.impedance / self.sound_speed

    @property
    def axial_pii(self) -> RealArray:
        """Pulse-intensity integral over one period on the axis [J/m^2], (nrec,)."""
        return iec.pulse_integrals(self.axis, self.frequency, self.impedance, axis=1)[1]

    def _level_metrics(self, level_db: float) -> dict[str, float]:
        """Focal length, widths and area at ``level_db`` below the focal value."""
        q = self.focal_quantity
        prof = self.axial(q)
        i = self.focal_index
        g = self.grid
        ix, iy = self.axis_ij
        level = 10 ** (level_db / 20) * prof[i]
        lines = {
            "p_positive": (self.xz_p_plus, self.yz_p_plus),
            "p_negative": (np.abs(self.xz_p_minus), np.abs(self.yz_p_minus)),
            "first_harmonic": (self.xz_amplitude, self.yz_amplitude),
        }[q]
        lvl = f"{abs(level_db):g}dB"
        out = {
            f"focal_length_{lvl}": width_at_level(self.z, prof, i, level),
            f"focal_width_x_{lvl}": width_at_level(g.x, lines[0][i], ix, level),
            f"focal_width_y_{lvl}": width_at_level(g.y, lines[1][i], iy, level),
        }
        if self.focal_plane is not None:
            out[f"focal_area_{lvl}"] = connected_area(self._focal_map(), (ix, iy), level, g.dx * g.dy)
        return out

    def _focal_map(self) -> RealArray:
        """The focal quantity over the focal plane."""
        assert self.focal_plane is not None
        maps = self.plane_maps(self.focal_plane)
        return {
            "p_positive": maps["p_positive"],
            "p_negative": -maps["p_negative"],
            "first_harmonic": maps["amplitude"][0],
        }[self.focal_quantity]

    def metrics(self, harmonics: int | None = None, level_db: float | None = None) -> dict[str, Any]:
        """Return the field metrics, with the IEC parameters under ``"iec"``.

        The focus is the plane recorded as such during the march: the maximum of the focal
        quantity on the beam axis. Widths, lengths and areas are at each of ``levels_db``
        (or at ``level_db``) relative to the focal value of the focal quantity, NaN when the
        region reaches a domain edge. ``docs/iec_outputs.md`` describes every entry.
        """
        harmonics = harmonics or self.metrics_harmonics
        levels = [level_db] if level_db is not None else list(self.levels_db)
        q = self.focal_quantity
        prof = self.axial(q)
        i = self.focal_index
        ix, iy = self.axis_ij
        g = self.grid
        intensity = self.axial_intensity
        p_r = float(-self.axis_p_minus[i])
        out: dict[str, Any] = {
            "focal_quantity": q,
            "focus": {
                "x": float(g.x[ix]),
                "y": float(g.y[iy]),
                "z": float(self.z[i]),
                "index": i,
                "on_edge": i in (0, self.z.size - 1),
                "p_positive": float(self.axis_p_plus[i]),
                "p_negative": float(self.axis_p_minus[i]),
                "harmonics": [float(a) for a in np.abs(self.axis[i, :harmonics])],
                "intensity": float(intensity[i]),
                "heating_rate": float(self.axial_heating[i]),
                "mechanical_index_water": p_r / 1e6 / math.sqrt(self.frequency / 1e6),
            },
            "axis_peak_p_positive": {
                "value": float(self.axis_p_plus.max()),
                "z": float(self.z[int(np.argmax(self.axis_p_plus))]),
            },
            "axis_peak_p_negative": {
                "value": float(self.axis_p_minus.min()),
                "z": float(self.z[int(np.argmin(self.axis_p_minus))]),
            },
            "axis_peak_intensity": {"value": float(intensity.max()), "z": float(self.z[int(np.argmax(intensity))])},
            "source_amplitude": self.source_amplitude,
            "focal_gain": float(abs(self.axis[i, 0]) / self.source_amplitude)
            if self.source_amplitude > 0
            else math.nan,
            "spatial_peak_intensity": {
                "value": float(self.plane_intensity.max()),
                "z": float(self.z[int(np.argmax(self.plane_intensity))]),
            },
            "power_source": float(self.power[0]),
            "power_focal_plane": float(self.power[i]),
            "power_end": float(self.power[-1]),
            "godunov_from_z": self.godunov_from,
        }
        for lv in levels:
            out.update(self._level_metrics(lv))
        if self.focal_plane is not None:
            out["focal_plane_peak_intensity"] = float(self.plane_maps(self.focal_plane)["intensity"].max())
        if self.focusing_depth > 0:
            out["focal_shift"] = float(self.z[i] - g.z0 - self.focusing_depth)
        if self.peak_index >= 0:
            pi, pj = self.peak_xy_index
            out["spatial_peak_p_positive"] = {
                "value": float(self.plane_p_plus[self.peak_index]),
                "x": float(g.x[pi]),
                "y": float(g.y[pj]),
                "z": float(self.z[self.peak_index]),
                "off_axis": (pi, pj) != (ix, iy),
            }
            out["spatial_peak_p_negative"] = {
                "value": float(np.nanmin(self.plane_p_minus)),
                "z": float(self.z[int(np.nanargmin(self.plane_p_minus))]),
            }
        out["iec"] = self.iec_metrics(prof)
        return out

    def iec_metrics(self, prof: RealArray | None = None) -> dict[str, Any]:
        """The IEC field parameters (see ``docs/iec_outputs.md`` and :mod:`pywave3d.iec`)."""
        g = self.grid
        i = self.focal_index
        ix, iy = self.axis_ij
        prof = self.axial() if prof is None else prof
        cell = g.dx * g.dy
        f0 = self.frequency
        focus_h = self.axis[i]
        freqs = iec.working_frequencies(focus_h, f0)
        f_awf = freqs["arithmetic_mean"] if math.isfinite(freqs["arithmetic_mean"]) else f0
        ppsi, pii = iec.pulse_integrals(focus_h, f0, self.impedance)
        i_spta = float(self.plane_intensity.max())
        out: dict[str, Any] = {
            "field": "continuous wave: pulse quantities are per period, and I_pa = I_ta",
            "acoustic_working_frequency": freqs,
            "p_c": float(self.axis_p_plus[i]),
            "p_r": float(-self.axis_p_minus[i]),
            "I_ta_focus": float(self.axial_intensity[i]),
            "I_spta": i_spta,
            "I_sppa": i_spta,
            "ppsi_focus": float(ppsi),
            "pii_focus": float(pii),
            "pii_spatial_peak": i_spta / f0,
            "beam_alignment_axis": (
                iec.beam_alignment_axis(self.z, self.centroid_x, self.centroid_y)
                if self.centroid_x is not None and self.centroid_y is not None
                else {}
            ),
            "side_lobes": {
                "axial": iec.secondary_maxima(self.z, prof, i),
                "x": iec.secondary_maxima(g.x, self.xz_p_plus[i], ix),
                "y": iec.secondary_maxima(g.y, self.yz_p_plus[i], iy),
            },
            "nonlinear": {
                "p_c_over_p_r": float(self.axis_p_plus[i] / -self.axis_p_minus[i])
                if self.axis_p_minus[i] < 0
                else math.nan,
                "P2_over_P1": float(abs(focus_h[1]) / abs(focus_h[0]))
                if focus_h.size > 1 and focus_h[0] != 0
                else math.nan,
                **iec.shock_parameters(focus_h, f0),
            },
        }

        # power: plane-wave estimate on every plane, exact (angular spectrum) on the kept ones
        exact = {}
        for name, plane in (("source", self.source_plane), ("focal_plane", self.focal_plane), ("end", self.final)):
            if plane is not None:
                iz = iec.active_intensity(plane, g, f0, self.sound_speed, self.density)
                exact[name] = float(iz.sum() * cell)
                if name == "focal_plane":
                    out["I_ta_focus_exact"] = float(iz[ix, iy])
                    out["I_spta_focal_plane_exact"] = float(iz.max())
                    lvl = 10 ** (self.levels_db[0] / 20) * prof[i]
                    fmap = self._focal_map()
                    from scipy.ndimage import label

                    labels, _ = label(fmap >= lvl)
                    region = labels == labels[ix, iy] if labels[ix, iy] else np.zeros_like(fmap, dtype=bool)
                    pw_i = self.intensity(plane)
                    tag = f"{abs(self.levels_db[0]):g}dB"
                    out[f"power_in_focal_area_{tag}"] = float(pw_i[region].sum() * cell)
                    out[f"power_in_focal_area_{tag}_exact"] = float(iz[region].sum() * cell)
        out["power"] = {
            "source": float(self.power[0]),
            "focal_plane": float(self.power[i]),
            "end": float(self.power[-1]),
            "exact": exact,
        }

        # hydrophone spatial averaging at the focus
        if self.hydrophone_diameter > 0 and self.focal_plane is not None:
            avg = iec.spatial_average(self.focal_plane, g, self.hydrophone_diameter)[:, ix, iy]
            w = waveform(avg.reshape(-1, 1), 16 * avg.size)[:, 0]
            out["hydrophone"] = {
                "diameter": self.hydrophone_diameter,
                "p_c": float(w.max()),
                "p_r": float(-w.min()),
                "harmonics": [float(a) for a in np.abs(avg[: self.metrics_harmonics])],
            }

        # derated exposure indices (IEC 62359)
        p_r_axis = -self.axis_p_minus
        if self.source_plane is not None:
            src_i = self.intensity(self.source_plane)
            a_aprt = float(np.count_nonzero(src_i >= 10 ** (-1.2) * src_i.max()) * cell) if src_i.max() > 0 else 0.0
        else:
            a_aprt = math.nan
        zrel = self.z - g.z0
        ti = iec.thermal_indices(zrel, float(self.power[0]), self.axial_intensity, f_awf, a_aprt)
        mi = iec.mechanical_index(zrel, p_r_axis, self.axial_pii, f_awf, z_bp=ti["z_bp"])
        d = iec.derating(f_awf, zrel, self.derating)
        out["derated"] = {
            "coefficient_dB_per_cm_MHz": self.derating,
            **mi,
            **ti,
            "I_spta_derated_axis": float((self.axial_intensity * d**2).max()),
            "p_r_derated_max_axis": float((p_r_axis * d).max()),
        }

        # physiotherapy (IEC 61689): beam cross-sectional area at face_distance, A_ER, R_BN
        if self.face_plane is not None:
            fi = self.intensity(self.face_plane)
            a_bcs = iec.area_holding(fi, cell, iec.BCS_FRACTION)
            a_er = iec.F_AC * a_bcs
            out["physiotherapy"] = {
                "z": float(self.z[self.face_index]),
                "A_BCS": a_bcs,
                "A_ER": a_er,
                "effective_intensity": float(self.power[0] / a_er) if a_er > 0 else math.nan,
                "R_BN": float(i_spta / (self.power[0] / a_er)) if a_er > 0 and self.power[0] > 0 else math.nan,
            }
        return out

    # -- files ---------------------------------------------------------------------------

    def save(self, directory: str | Path = ".", *, harmonics: int = 4) -> list[Path]:
        """Write the outputs and return the files written:

        * ``axial.txt``: the beam axis: z, p+, p-, intensity, heating rate, |P_1| .. |P_h|;
        * ``focal_plane.txt``: the focal plane: x, y, p+, p-, intensity, |P_1| .. |P_h|;
        * ``metrics.json``: :meth:`metrics`, including the focus and the spatial peak;
        * ``results.npz``: every recorded array (axis, lines, planes, power, ...);
        * one waveform file per requested point.
        """
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        h = min(harmonics, self.axis.shape[1])
        amp_cols = "  ".join(f"|P{n + 1}| [Pa]" for n in range(h))
        written = [d / "axial.txt", d / "metrics.json", d / "results.npz"]
        der = iec.derating(self.frequency, self.z - self.grid.z0, self.derating)
        cols = [
            self.z,
            self.axis_p_plus,
            self.axis_p_minus,
            self.axial_intensity,
            self.axial_heating,
            self.axial_pii,
            -self.axis_p_minus * der,
            self.axial_intensity * der**2,
            *np.abs(self.axis[:, :h]).T,
        ]
        np.savetxt(
            written[0],
            np.column_stack(cols),
            fmt="%.8e",
            header=f"beam axis (x, y) = {self.axis_xy}; focus at z = {self.z[self.focal_index]:.6e} m; "
            f"derating {self.derating} dB/(cm MHz) at the base frequency\n"
            f"z [m]  p+ [Pa]  p- [Pa]  I [W/m^2]  q [W/m^3]  pii [J/m^2]  p_r,a [Pa]  I_ta,a [W/m^2]  {amp_cols}",
        )
        cx = self.centroid_x if self.centroid_x is not None else np.full(self.z.size, np.nan)
        cy = self.centroid_y if self.centroid_y is not None else np.full(self.z.size, np.nan)
        path = d / "planes.txt"
        np.savetxt(
            path,
            np.column_stack([self.z, self.power, self.plane_p_plus, self.plane_p_minus, self.plane_intensity, cx, cy]),
            fmt="%.8e",
            header="every recorded plane: z [m]  power [W]  max p+ [Pa]  min p- [Pa]  max I [W/m^2]  "
            "beam centre x [m]  y [m] (intensity centroid of the -6 dB region)",
        )
        written.append(path)
        with written[1].open("w", encoding="utf-8") as f:
            json.dump(self.metrics(harmonics), f, indent=2, allow_nan=True)
        arrays: dict[str, Any] = {k: v for k, v in vars(self).items() if isinstance(v, np.ndarray)}
        np.savez_compressed(written[2], x=self.grid.x, y=self.grid.y, **arrays)
        if self.focal_plane is not None:
            maps = self.plane_maps(self.focal_plane)
            gx, gy = np.meshgrid(self.grid.x, self.grid.y, indexing="ij")
            cols = [gx, gy, maps["p_positive"], maps["p_negative"], maps["intensity"], *maps["amplitude"][:h]]
            path = d / "focal_plane.txt"
            np.savetxt(
                path,
                np.column_stack([c.ravel() for c in cols]),
                fmt="%.8e",
                header=f"focal plane z = {self.z[self.focal_index]:.6e} m "
                f"({self.focal_quantity} maximum on the beam axis)\n"
                f"x [m]  y [m]  p+ [Pa]  p- [Pa]  I [W/m^2]  {amp_cols}",
            )
            written.append(path)
        for name, (pos, harm) in self.points.items():
            t, p = self.waveform_at(harm)
            path = d / name
            np.savetxt(
                path,
                np.column_stack([t, p, p**2 / self.impedance]),
                header=f"t [s]  p [Pa]  instantaneous intensity p^2/(rho c) [W/m^2] at (x, y, z) = {pos}",
                fmt="%.8e",
            )
            written.append(path)
        return written
