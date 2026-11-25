"""Medium properties and the grid."""

import math

import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation

import pywave3d as pw


def test_water_defaults(water: pw.Medium) -> None:
    assert water.sound_speed == 1482.0
    assert water.attenuation.law == "thermoviscous"
    # delta omega^2 / (2 c0^3) = 0.0263 Np/m (0.228 dB/m) at 1 MHz; measured 0.025 Np/m (Pinkerton)
    assert water.alpha(1e6, np.array([1.0]))[0] == pytest.approx(0.0263, rel=0.01)
    assert water.epsilon == pytest.approx(3.5 / (998 * 1482**3))


def test_thermoviscous_absorption_is_quadratic(water: pw.Medium) -> None:
    n = np.arange(1.0, 9.0)
    a = water.alpha(1e6, n)
    np.testing.assert_allclose(a / a[0], n**2)
    assert water.alpha(2e6, np.array([1.0]))[0] == pytest.approx(a[1])
    np.testing.assert_array_equal(water.dispersion(n), 0.0)


def test_power_law_and_dispersion() -> None:
    m = pw.Medium(attenuation=pw.Attenuation("power_law", alpha0=5.0, power=1.5, dispersion=True))
    n = np.arange(1.0, 5.0)
    np.testing.assert_allclose(m.alpha(1e6, n), 5.0 * n**1.5)
    dk = m.dispersion(n)
    assert dk[0] == 0.0
    assert np.all(dk[1:] < 0)  # 1 < y < 2: the phase speed rises with frequency
    m1 = pw.Medium(attenuation=pw.Attenuation("power_law", alpha0=5.0, power=1.0, dispersion=True))
    np.testing.assert_allclose(m1.dispersion(n), -(2 / math.pi) * 5.0 * n * np.log(n))


def test_invalid_media() -> None:
    with pytest.raises(ValueError):
        pw.Medium(sound_speed=-1)
    with pytest.raises((ValueError, BeartypeCallHintParamViolation)):
        pw.Attenuation("cubic")  # type: ignore[arg-type]


def test_shock_distance(water: pw.Medium) -> None:
    xs = water.shock_distance(1e6, 1e6)
    assert xs == pytest.approx(998 * 1482**3 / (3.5 * 2 * math.pi * 1e6 * 1e6))
    assert math.isinf(pw.Medium(nonlinearity=0).shock_distance(1e6, 1e6))


def test_grid_from_lengths() -> None:
    g = pw.Grid.from_lengths(0.01, 0.012, 0.02, wavelength=1.5e-3, points_per_wavelength=(4, 4, 5), nt=15)
    assert g.nx % 2 == 0 and g.ny % 2 == 0 and g.nt == 16
    lx, ly, lz = g.lengths
    assert (lx, ly, lz) == pytest.approx((0.01, 0.012, 0.02))
    assert g.x[-1] == pytest.approx(0.01) and g.z[-1] == pytest.approx(0.02)
    assert g.n_harmonics == 8 and g.n_steps == g.nz - 1
    np.testing.assert_allclose(g.kx, 2 * np.pi * np.fft.fftfreq(g.nx, g.dx))


def test_grid_nearest_and_centre(small_grid: pw.Grid) -> None:
    assert small_grid.centre_index == (16, 16)
    assert small_grid.nearest(*small_grid.centre) == (16, 16)
    assert small_grid.nearest(-1.0, 1.0, 1.0) == (0, 31, 20)


def test_grid_validation() -> None:
    with pytest.raises(ValueError):
        pw.Grid(nx=1, ny=4, nz=4, nt=8, dx=1e-3, dy=1e-3, dz=1e-3)
    with pytest.raises(ValueError):
        pw.Grid(nx=4, ny=4, nz=4, nt=7, dx=1e-3, dy=1e-3, dz=1e-3)


def test_fast_fft_sizes() -> None:
    g = pw.Grid.from_lengths(0.04, 0.04, 0.01, wavelength=1.482e-3, points_per_wavelength=(6, 6, 6))
    assert (g.nx, g.ny) == (168, 168)  # not 164 = 4 x 41
    s = pw.Grid.from_lengths(0.04, 0.04, 0.01, wavelength=1.482e-3, points_per_wavelength=(6, 6, 6), fast_sizes=False)
    assert s.nx == 164
    e = pw.Grid.from_lengths(0.04, 0.04, 0.01, wavelength=1.482e-3, number_of_points=(41, 41, 9))
    assert e.nx == 42  # explicit sizes are only made even
