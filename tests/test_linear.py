"""One linear step against its closed form."""

import numpy as np
import pytest

import pywave3d as pw

F0 = 1e6


def mode(grid: pw.Grid, mx: int, my: int, k: int) -> np.ndarray:
    """Harmonic 1 holds the transverse Fourier mode (mx, my); shape (K, nx, ny)."""
    f = np.zeros((k, grid.nx, grid.ny), dtype=complex)
    f[0] = np.exp(1j * (grid.kx[mx] * grid.x[:, None] + grid.ky[my] * grid.y[None, :]))
    return f


def test_plane_wave_only_absorbed(small_grid: pw.Grid, water: pw.Medium) -> None:
    lin = pw.LinearPropagator(small_grid, water, F0)
    f = np.ones((small_grid.n_harmonics, small_grid.nx, small_grid.ny), dtype=complex)
    lin.apply(f)
    alpha = water.alpha(F0, np.arange(1.0, small_grid.n_harmonics + 1))
    np.testing.assert_allclose(f[:, 3, 5], np.exp(-alpha * small_grid.dz), rtol=1e-12)


def test_transverse_mode_phase(small_grid: pw.Grid, water: pw.Medium) -> None:
    lin = pw.LinearPropagator(small_grid, water, F0, absorption=False)
    f = mode(small_grid, 2, 1, small_grid.n_harmonics)
    f0 = f.copy()
    lin.apply(f)
    k = 2 * np.pi * F0 / water.sound_speed
    kt2 = small_grid.kx[2] ** 2 + small_grid.ky[1] ** 2
    expected = np.exp(-1j * (np.sqrt(k**2 - kt2) - k) * small_grid.dz)
    np.testing.assert_allclose(f[0], f0[0] * expected, rtol=1e-10, atol=1e-12)


def test_evanescent_mode_decays(water: pw.Medium) -> None:
    g = pw.Grid(nx=16, ny=16, nz=5, nt=2, dx=2e-4, dy=2e-4, dz=2e-4)  # kx up to 8 k
    lin = pw.LinearPropagator(g, water, F0, absorption=False)
    f = mode(g, 6, 0, 1)
    lin.apply(f)
    k = 2 * np.pi * F0 / water.sound_speed
    decay = np.exp(-np.sqrt(g.kx[6] ** 2 - k**2) * g.dz)
    np.testing.assert_allclose(np.abs(f[0]), decay, rtol=1e-10)


def test_half_steps_make_a_step(small_grid: pw.Grid, water: pw.Medium) -> None:
    lin = pw.LinearPropagator(small_grid, water, F0)
    rng = np.random.default_rng(1)
    f = rng.standard_normal((small_grid.n_harmonics, 32, 32)) + 1j * rng.standard_normal(
        (small_grid.n_harmonics, 32, 32)
    )
    a, b = f.copy(), f.copy()
    lin.apply(a, 1.0)
    lin.apply(b, 0.5)
    lin.apply(b, 0.5)
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError):
        lin.apply(a, 0.3)


def test_paraxial_phase(small_grid: pw.Grid, water: pw.Medium) -> None:
    lin = pw.LinearPropagator(small_grid, water, F0, absorption=False, paraxial=True)
    f = mode(small_grid, 1, 0, small_grid.n_harmonics)
    f0 = f.copy()
    lin.apply(f)
    k = 2 * np.pi * F0 / water.sound_speed
    expected = np.exp(1j * small_grid.kx[1] ** 2 / (2 * k) * small_grid.dz)
    np.testing.assert_allclose(f[0], f0[0] * expected, rtol=1e-10, atol=1e-12)


def test_active_harmonics_only(small_grid: pw.Grid, water: pw.Medium) -> None:
    lin = pw.LinearPropagator(small_grid, water, F0)
    f = np.ones((small_grid.n_harmonics, 32, 32), dtype=complex)
    lin.apply(f, n_active=1)
    assert np.all(f[1:] == 1)  # untouched


def test_absorption_only_without_diffraction(small_grid: pw.Grid, water: pw.Medium) -> None:
    lin = pw.LinearPropagator(small_grid, water, F0, diffraction=False)
    f = mode(small_grid, 3, 3, small_grid.n_harmonics)
    f0 = f.copy()
    lin.apply(f)
    np.testing.assert_allclose(f[0], f0[0] * np.exp(-water.alpha(F0, np.array([1.0]))[0] * small_grid.dz))


def test_absorbing_layer(small_grid: pw.Grid, water: pw.Medium) -> None:
    layer = pw.AbsorbingLayer(width=2e-3, strength=1e3)
    sigma = layer.profile(small_grid)
    assert sigma is not None
    assert sigma[16, 16] == 0 and sigma[0, 16] == pytest.approx(1e3) and sigma[16, -1] == pytest.approx(1e3)
    assert pw.AbsorbingLayer().profile(small_grid) is None
    lin = pw.LinearPropagator(small_grid, water, F0, diffraction=False, absorption=False, layer=layer)
    f = np.ones((small_grid.n_harmonics, 32, 32), dtype=complex)
    lin.apply(f)
    np.testing.assert_allclose(f[0].real, np.exp(-sigma * small_grid.dz))


def test_single_precision(small_grid: pw.Grid, water: pw.Medium) -> None:
    lin = pw.LinearPropagator(small_grid, water, F0, dtype=np.complex64)
    f = mode(small_grid, 1, 1, small_grid.n_harmonics).astype(np.complex64)
    lin.apply(f)
    assert f.dtype == np.complex64


def test_layer_settings() -> None:
    with pytest.raises(ValueError):
        pw.AbsorbingLayer(width=1e-3, strength=1.0, kind="cpml")
    with pytest.raises(ValueError):
        pw.AbsorbingLayer(width=1e-3, strength=1.0, kind="pml", levels=0)
    layer = pw.AbsorbingLayer.for_reflection(4e-3, 1e-4, power=2)
    assert layer.kind == "pml"
    # round trip through the layer at normal incidence: exp(-2 int sigma) = R
    assert np.exp(-2 * layer.strength * 4e-3 / 3) == pytest.approx(1e-4)


def test_pml_leaves_a_plane_wave_alone(small_grid: pw.Grid, water: pw.Medium) -> None:
    """kx = ky = 0 does not move sideways, so the PML does not touch it (the sponge does)."""
    pml = pw.AbsorbingLayer(width=2e-3, strength=1e4, kind="pml")
    lin = pw.LinearPropagator(small_grid, water, F0, absorption=False, layer=pml)
    f = np.ones((small_grid.n_harmonics, 32, 32), dtype=complex)
    lin.apply(f)
    np.testing.assert_allclose(f, 1.0, atol=1e-12)


@pytest.mark.parametrize("paraxial", [False, True])
def test_pml_decay_of_a_transverse_mode(small_grid: pw.Grid, water: pw.Medium, paraxial: bool) -> None:
    """At the outer edge (sigma = strength) a mode (kx, 0) decays by exp(-sigma kx^2 dz / (k kz))."""
    pml = pw.AbsorbingLayer(width=2e-3, strength=500.0, kind="pml", levels=3)
    lin = pw.LinearPropagator(small_grid, water, F0, absorption=False, paraxial=paraxial, layer=pml)
    free = pw.LinearPropagator(small_grid, water, F0, absorption=False, paraxial=paraxial)
    f = mode(small_grid, 3, 0, 1)
    g = f.copy()
    lin.apply(f, 0.5)
    free.apply(g, 0.5)
    k = 2 * np.pi * F0 / water.sound_speed
    kx = small_grid.kx[3]
    kz = k if paraxial else np.sqrt(k**2 - kx**2)
    expected = np.exp(-500.0 * kx**2 / (k * kz) * small_grid.dz / 2)
    np.testing.assert_allclose(f[0, 0] / g[0, 0], expected, rtol=1e-10)  # x = x0: the edge
    np.testing.assert_allclose(f[0, 16], g[0, 16], rtol=1e-12)  # outside the layer: untouched


def test_pml_without_diffraction_does_nothing(small_grid: pw.Grid, water: pw.Medium) -> None:
    pml = pw.AbsorbingLayer(width=2e-3, strength=1e3, kind="pml")
    lin = pw.LinearPropagator(small_grid, water, F0, diffraction=False, absorption=False, layer=pml)
    f = np.ones((small_grid.n_harmonics, 32, 32), dtype=complex)
    lin.apply(f)
    np.testing.assert_array_equal(f, 1.0)


def _beam(water: pw.Medium, nx: int, layer: pw.AbsorbingLayer | None, theta: float, x0: float, nz: int) -> np.ndarray:
    """A Gaussian beam tilted by theta in x, after nz steps, on nx x 128 points of lambda/4."""
    lam = water.sound_speed / F0
    dx, ny = lam / 4, 128
    g = pw.Grid(nx=nx, ny=ny, nz=nz, nt=2, dx=dx, dy=dx, dz=lam / 2, x0=-nx * dx / 2, y0=-ny * dx / 2)
    lin = pw.LinearPropagator(g, water, F0, absorption=False, layer=layer)
    x, y = g.x[:, None], g.y[None, :]
    f = np.exp(-((x - x0) ** 2 + y**2) / 1.5e-3**2 - 2j * np.pi / lam * np.sin(theta) * x)[None].astype(complex)
    for _ in range(nz):
        lin.apply(f)
    return f[0]


def test_pml_against_wrap_around(water: pw.Medium) -> None:
    """A beam leaving through the side, against the same beam on a domain 5 times wider."""
    n, width = 64, 16 * water.sound_speed / F0 / 4
    inner = (slice(16, n - 16), slice(None))
    sponge = pw.AbsorbingLayer.for_reflection(width, 1e-4, kind="sponge")
    pml = pw.AbsorbingLayer.for_reflection(width, 1e-4, kind="pml")

    def errors(theta: float, x0: float, nz: int) -> dict[str, float]:
        ref = _beam(water, 5 * n, None, theta, x0, nz)[2 * n : 3 * n]
        out = {}
        for name, layer in (("none", None), ("sponge", sponge), ("pml", pml)):
            out[name] = float(np.abs(_beam(water, n, layer, theta, x0, nz) - ref)[inner].max())
        return out

    # oblique beam at 25 degrees: the PML absorbs it better than the sponge
    e = errors(np.radians(25), 0.0, 60)
    assert e["pml"] < 0.6 * e["sponge"] and e["pml"] < 0.1 * e["none"]
    # a beam along z next to the edge: the PML hardly touches it, the sponge eats into it
    e = errors(0.0, 6e-3, 40)
    assert e["pml"] < 0.1 * e["sponge"]


def test_shock_smoothing_profile() -> None:
    f = 0.05
    sm = pw.AxialSmoothing(focal_length=f, peak_multiplier=20.0, post_focal_multiplier=3.0)
    m = sm.multiplier(np.array([0.0, 0.995 * f, 2 * f]))
    np.testing.assert_allclose(m, [1.0, 20.0, 3.0], rtol=2e-3)  # tanh(3.5) at the centre
    # the paper's form: a1 + (A - 1)/2 [tanh - a2 tanh], a1 = (1 + B)/2, a2 = 1 + (1 - B)/(A - 1)
    a, b = 20.0, 3.0
    z = np.linspace(0.9 * f, 1.1 * f, 101)
    paper = (1 + b) / 2 + (a - 1) / 2 * (
        np.tanh((z - 0.96 * f) / (0.01 * f)) - (1 + (1 - b) / (a - 1)) * np.tanh((z - 1.03 * f) / (0.01 * f))
    )
    np.testing.assert_allclose(sm.multiplier(z), paper, rtol=1e-12)
    # the step integral is exact, also over a step much longer than the widths
    za, zb = 0.9 * f, 1.2 * f
    zz = np.linspace(za, zb, 200001)
    mm = sm.multiplier(zz) - 1
    np.testing.assert_allclose(sm.excess(za, zb), np.sum((mm[1:] + mm[:-1]) / 2 * np.diff(zz)), rtol=1e-8)
    assert pw.AxialSmoothing(focal_length=f, peak_multiplier=1.0).excess(0.0, 2 * f) == 0.0
    with pytest.raises(ValueError):
        pw.AxialSmoothing(start=1.1, end=1.0)
    with pytest.raises(ValueError):
        pw.AxialSmoothing().multiplier(0.0)  # no focal length


def test_shock_smoothing_scales_the_diffusivity(small_grid: pw.Grid, water: pw.Medium) -> None:
    f = 0.05
    sm = pw.AxialSmoothing(focal_length=f, peak_multiplier=7.0)
    lin = pw.LinearPropagator(small_grid, water, F0, smoothing=sm)
    shape = (small_grid.n_harmonics, small_grid.nx, small_grid.ny)
    alpha = water.alpha(F0, np.arange(1.0, small_grid.n_harmonics + 1))
    # well before the focus: the physical absorption only
    p = np.ones(shape, dtype=complex)
    lin.apply(p, z=small_grid.z0)
    np.testing.assert_allclose(p[:, 3, 5], np.exp(-alpha * small_grid.dz), rtol=1e-9)
    # inside the peak: seven times the diffusivity
    p = np.ones(shape, dtype=complex)
    lin.apply(p, z=small_grid.z0 + 0.995 * f)
    np.testing.assert_allclose(p[:, 3, 5], np.exp(-7 * alpha * small_grid.dz), rtol=1e-3)
    with pytest.raises(ValueError):
        lin.apply(p)  # the position is needed
