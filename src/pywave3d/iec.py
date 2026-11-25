"""Field parameters of the IEC standards, computed from the harmonics of the march.

The functions are pure: they take harmonics (K, ...), profiles or planes and return numbers or
arrays in SI units. :meth:`pywave3d.results.Results.metrics` uses them for the ``iec`` section
of ``metrics.json``; ``docs/iec_outputs.md`` lists which standard asks for what.

The definitions are paraphrased from the standards (IEC 62127-1, 61828, TS 62556, 62359 and
61689); constants and formulae that come from a standard are marked, and should be checked
against the edition in force before results are declared.

The field is periodic (continuous wave): pulse quantities are per period, and the
pulse-average intensity equals the temporal-average intensity.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.fft as sfft

from pywave3d._types import ComplexArray, RealArray
from pywave3d.grid import Grid
from pywave3d.nonlinear import waveform

#: IEC 62359 derating coefficient [dB / (cm MHz)]
DERATING_DB_PER_CM_MHZ = 0.3
#: IEC 62359 constants of the thermal indices
C_TIS = 210.0  # mW MHz
C_TIB1 = 50.0  # mW / cm
C_TIB2 = 4.4  # mW
C_TIC = 40.0  # mW / cm
#: IEC 61689: effective radiating area = F_AC x beam cross-sectional area at 0.3 cm
F_AC = 1.333
#: IEC 61689: the beam cross-sectional area holds this fraction of the mean-square pressure
BCS_FRACTION = 0.75


# -- derating -----------------------------------------------------------------------------


def derating(frequency: float, z: RealArray, coefficient: float = DERATING_DB_PER_CM_MHZ) -> RealArray:
    """Pressure derating factor 10^(-a f z / 20) (f in MHz, z in cm), IEC 62359.

    Square it for intensities, pulse-intensity integrals and power.
    """
    return 10 ** (-coefficient * (frequency / 1e6) * (np.asarray(z, dtype=float) * 100) / 20)


# -- frequencies --------------------------------------------------------------------------


def working_frequencies(harmonics: ComplexArray, frequency: float) -> dict[str, float]:
    """Acoustic working frequency of a periodic waveform at a point (IEC 62127-1).

    * ``arithmetic_mean``: (f1 + f2) / 2, with f1 and f2 the lowest and highest frequencies
      at which the (line) spectrum is within 3 dB of its maximum;
    * ``zero_crossing``: (number of zero crossings per period) / 2 x f0.
    """
    amp = np.abs(harmonics)
    if amp.max() == 0:
        return {"arithmetic_mean": math.nan, "zero_crossing": math.nan}
    within = np.nonzero(amp >= amp.max() / math.sqrt(2))[0] + 1
    w = waveform(harmonics.reshape(-1, 1), 16 * harmonics.shape[0])[:, 0]
    s = np.signbit(w)
    crossings = int(np.count_nonzero(s != np.roll(s, 1)))
    return {
        "arithmetic_mean": float((within[0] + within[-1]) / 2 * frequency),
        "zero_crossing": float(crossings / 2 * frequency),
    }


# -- intensity and power ------------------------------------------------------------------


def pulse_integrals(
    harmonics: ComplexArray, frequency: float, impedance: float, axis: int = 0
) -> tuple[RealArray, RealArray]:
    """Pulse-pressure-squared integral [Pa^2 s] and pulse-intensity integral [J/m^2] over one
    period, by Parseval: ppsi = T sum |P_n|^2 / 2, pii = ppsi / (rho0 c0)."""
    ppsi = np.asarray((np.abs(harmonics) ** 2).sum(axis=axis) / (2 * frequency), dtype=float)
    return ppsi, np.asarray(ppsi / impedance)


def active_intensity(
    harmonics: ComplexArray, grid: Grid, frequency: float, sound_speed: float, density: float
) -> RealArray:
    """Exact time-averaged axial intensity I_z = sum_n Re(P_n v_z,n*) / 2 over a plane (nx, ny).

    The axial particle velocity comes from the angular spectrum: each propagating plane-wave
    component of harmonic n has v_z = P kz_n / (rho0 omega_n); evanescent ones carry no power.
    The plane-wave estimate sum |P_n|^2 / (2 rho0 c0) is larger by the obliquity factor.
    """
    k = harmonics.shape[0]
    kt2 = grid.kx[:, None] ** 2 + grid.ky[None, :] ** 2
    out = np.zeros((grid.nx, grid.ny))
    for n in range(k):
        if not np.any(harmonics[n]):
            continue
        kn = 2 * math.pi * frequency * (n + 1) / sound_speed
        ratio = np.sqrt(np.clip(1 - kt2 / kn**2, 0, None))  # kz / k_n, 0 if evanescent
        v = sfft.ifft2(sfft.fft2(harmonics[n]) * ratio) / (density * sound_speed)
        out += 0.5 * (harmonics[n] * np.conj(v)).real
    return out


def area_holding(intensity: RealArray, cell: float, fraction: float) -> float:
    """Smallest area [m^2] that holds ``fraction`` of the plane's total of ``intensity``."""
    v = np.sort(intensity.ravel())[::-1]
    total = v.sum()
    if total <= 0:
        return 0.0
    n = int(np.searchsorted(np.cumsum(v), fraction * total)) + 1
    return float(min(n, v.size) * cell)


def spatial_average(harmonics: ComplexArray, grid: Grid, diameter: float) -> ComplexArray:
    """Harmonics averaged over a disc of ``diameter`` around every point, as a hydrophone of
    that sensitive-element diameter measures them (circular convolution on the grid)."""
    if diameter <= 0:
        return harmonics
    xc, yc = grid.centre
    x = grid.x - xc
    y = grid.y - yc
    disc = (np.hypot(x[:, None], y[None, :]) <= diameter / 2).astype(float)
    disc /= disc.sum()
    # the disc is centred on index (nx/2, ny/2): move it to (0, 0) before convolving
    kernel = sfft.fft2(np.roll(disc, (-grid.centre_index[0], -grid.centre_index[1]), axis=(0, 1)))
    return sfft.ifft2(sfft.fft2(harmonics, axes=(1, 2)) * kernel[None], axes=(1, 2))


# -- beam shape ---------------------------------------------------------------------------


def secondary_maxima(coord: RealArray, values: RealArray, index: int, lobe_db: float = -6.0) -> dict[str, float]:
    """Largest maxima outside the main lobe around ``index``, on either side.

    The main lobe extends on each side until the profile has fallen below ``lobe_db``
    (relative to ``values[index]``) and then reached a local minimum, so ripples on the main
    lobe are not counted as side lobes. Levels are amplitude ratios (20 log10), for pressure
    profiles. Returns, for ``before`` and ``after`` (lower and higher ``coord``), the value,
    its position and its level in dB (NaN if there is no secondary maximum on that side).
    """
    v = np.asarray(values, dtype=float)
    peak = v[index]
    level = 10 ** (lobe_db / 20) * peak
    lo = index
    while lo > 0 and (v[lo] >= level or v[lo - 1] <= v[lo]):
        lo -= 1
    hi = index
    while hi < v.size - 1 and (v[hi] >= level or v[hi + 1] <= v[hi]):
        hi += 1
    out: dict[str, float] = {}
    for side, seg, off, edge in (("before", v[:lo], 0, v[lo]), ("after", v[hi + 1 :], hi + 1, v[hi])):
        # a maximum needs a rise into it: a monotonic run to the domain edge has none
        if seg.size and peak > 0 and seg.max() > edge:
            j = int(np.argmax(seg)) + off
            out[f"{side}_value"] = float(v[j])
            out[f"{side}_position"] = float(coord[j])
            out[f"{side}_level_dB"] = float(20 * math.log10(v[j] / peak)) if v[j] > 0 else -math.inf
        else:
            out[f"{side}_value"] = math.nan
            out[f"{side}_position"] = math.nan
            out[f"{side}_level_dB"] = math.nan
    return out


def beam_alignment_axis(z: RealArray, cx: RealArray, cy: RealArray) -> dict[str, float]:
    """Least-squares line x = x0 + ax z, y = y0 + ay z through the beam centroids.

    Returns the intercepts at the first plane, the slopes and the tilt angle [rad] from z.
    """
    ok = np.isfinite(cx) & np.isfinite(cy)
    if ok.sum() < 2:
        return {"x0": math.nan, "y0": math.nan, "slope_x": math.nan, "slope_y": math.nan, "tilt": math.nan}
    zz = z[ok] - z[0]
    ax, bx = np.polyfit(zz, cx[ok], 1)
    ay, by = np.polyfit(zz, cy[ok], 1)
    return {
        "x0": float(bx),
        "y0": float(by),
        "slope_x": float(ax),
        "slope_y": float(ay),
        "tilt": float(math.atan(math.hypot(ax, ay))),
    }


# -- waveform shape -----------------------------------------------------------------------


def shock_parameters(harmonics: ComplexArray, frequency: float) -> dict[str, float]:
    """Steepness and shock front of the periodic waveform at a point.

    * ``steepness``: max dp/dt divided by omega |P_1| (1 for a sinusoid, large for a shock);
    * ``shock_amplitude``: the rise of the pressure across its steepest part, over one
      resolvable interval T / K (the width of a shock on K harmonics);
    * ``shocked``: steepness above 3 and a rise above |P_1| (a developed shock front).
    """
    k = harmonics.shape[0]
    nt = 16 * k
    w = waveform(harmonics.reshape(-1, 1), nt)[:, 0]
    omega = 2 * math.pi * frequency
    a1 = abs(harmonics[0])
    if a1 == 0:
        return {"steepness": math.nan, "shock_amplitude": 0.0, "shocked": False}
    dpdt = (np.roll(w, -1) - np.roll(w, 1)) * nt * frequency / 2
    half = max(1, nt // (2 * k))
    j = int(np.argmax(dpdt))
    rise = float(w[(j + half) % nt] - w[(j - half) % nt])
    steep = float(dpdt[j] / (omega * a1))
    return {"steepness": steep, "shock_amplitude": rise, "shocked": bool(steep > 3 and rise > a1)}


# -- exposure indices (IEC 62359) ---------------------------------------------------------


def mechanical_index(z: RealArray, p_r: RealArray, pii: RealArray, f_awf: float, z_bp: float = 0.0) -> dict[str, float]:
    """MI = p_r,alpha(z_MI) [MPa] / sqrt(f_awf [MHz]), at the depth z_MI (>= z_bp) of the
    maximum of the derated pulse-intensity integral (IEC 62359)."""
    d = derating(f_awf, z)
    pii_a = pii * d**2
    ok = z >= z_bp
    if not np.any(ok):
        return {"MI": math.nan, "z_MI": math.nan, "p_r_derated": math.nan}
    i = int(np.flatnonzero(ok)[np.argmax(pii_a[ok])])
    pra = float(p_r[i] * d[i])
    return {"MI": pra / 1e6 / math.sqrt(f_awf / 1e6), "z_MI": float(z[i]), "p_r_derated": pra}


def thermal_indices(
    z: RealArray, power: float, intensity_axis: RealArray, f_awf: float, aperture_area: float
) -> dict[str, float]:
    """Unscanned-mode thermal indices of IEC 62359 from the output power, the axial
    temporal-average intensity and the -12 dB output beam area (SI in, mW / cm inside).

    * TIS: P f / C_TIS for an aperture <= 1 cm^2; otherwise
      max over z >= z_bp of min(P_alpha(z), I_ta,alpha(z) x 1 cm^2) f / C_TIS;
    * TIB: max over z >= z_bp of min(sqrt(P_alpha I_ta,alpha) / C_TIB1, P_alpha / C_TIB2);
    * TIC: P / (D_eq C_TIC), D_eq = sqrt(4 A_aprt / pi);
    with z_bp = 1.5 D_eq and f in MHz.
    """
    f = f_awf / 1e6
    p_mw = power * 1e3
    a_cm2 = aperture_area * 1e4
    d_eq = math.sqrt(4 * a_cm2 / math.pi)  # cm
    z_bp = 1.5 * d_eq / 100  # m
    d2 = derating(f_awf, z) ** 2
    pa = p_mw * d2  # mW
    ia = intensity_axis * d2 * 1e3 / 1e4  # mW / cm^2
    ok = z >= z_bp
    if a_cm2 <= 1.0:
        tis = p_mw * f / C_TIS
    else:
        tis = float(np.max(np.minimum(pa, ia * 1.0)[ok]) * f / C_TIS) if np.any(ok) else math.nan
    tib = float(np.max(np.minimum(np.sqrt(pa * ia) / C_TIB1, pa / C_TIB2)[ok])) if np.any(ok) else math.nan
    tic = p_mw / (d_eq * C_TIC) if d_eq > 0 else math.nan
    return {"TIS": float(tis), "TIB": tib, "TIC": float(tic), "D_eq": d_eq / 100, "z_bp": z_bp, "A_aprt": aperture_area}
