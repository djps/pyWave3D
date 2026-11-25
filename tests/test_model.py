"""The model: reading options, the source files and the transducer."""

import math

import numpy as np
import pytest

import pywave3d as pw
from pywave3d.options import Options


def test_defaults_are_water_and_focused(water: pw.Medium) -> None:
    m = pw.WaveModel.from_options(Options({}))
    assert m.medium == water
    assert m.source is None
    assert m.transducer.focusing_depth == 0.05
    g = m.make_grid()
    assert g.nt == 32
    f = m.initial_field(g)
    assert f.shape == (16, g.nx, g.ny)
    assert np.all(f[1:] == 0)


def test_transducer_amplitude_from_power(water: pw.Medium) -> None:
    t = pw.Transducer(outer_radius=0.01, power=10.0)
    p0 = t.source_amplitude(water)
    assert 10.0 == pytest.approx(p0**2 * math.pi * 0.01**2 / (2 * water.impedance))
    assert pw.Transducer(amplitude=3.0).source_amplitude(water) == 3.0
    with pytest.raises(ValueError):
        pw.Transducer(outer_radius=0.01, focusing_depth=0.005)


def test_focused_source_phase(small_grid: pw.Grid, water: pw.Medium) -> None:
    t = pw.Transducer(outer_radius=3e-3, focusing_depth=0.02, amplitude=1.0)
    f = t.field(small_grid, water, 1e6, small_grid.centre)
    assert abs(f[16, 16]) == pytest.approx(1.0)
    assert f[0, 0] == 0
    # converging: the phase rises off axis, exp(+i k (sqrt(r^2 + d^2) - d))
    k = 2 * math.pi * 1e6 / water.sound_speed
    r = 4 * small_grid.dx
    assert np.angle(f[20, 16]) == pytest.approx(k * (math.hypot(r, 0.02) - 0.02))


def test_reads_the_reference_text_input(repo) -> None:
    opts = Options.load(repo / "options" / "woptions.json")
    m = pw.WaveModel.from_options(opts.child("Model"))
    assert m.source is not None
    assert m.source.harmonics.shape == (1, 50, 50)
    assert m.medium.sound_speed == 1487.0 and m.medium.diffusivity == 1.5e-6
    assert m.lengths == (0.01, 0.01, 0.02)
    g = m.make_grid(opts.child("Solver.DiffractSolver"))
    assert g.nt == 8
    f = m.initial_field(g)
    assert np.abs(f[0]).max() > 0


def test_bilinear_interpolation_is_exact_for_bilinear_data(tmp_path, small_grid: pw.Grid) -> None:
    x = np.linspace(0, small_grid.x[-1], 7)
    y = np.linspace(0, small_grid.y[-1], 9)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    vals = (1 + 2 * xx + 3 * yy + 4 * xx * yy * 1e3) * (1 + 0.5j)
    path = tmp_path / "pxy.txt"
    rows = np.column_stack([xx.ravel(), yy.ravel(), vals.real.ravel(), vals.imag.ravel()])
    np.savetxt(path, rows[::-1], header="x y re im")  # any order
    src = pw.Source.from_text(path)
    out = src.on_grid(small_grid, 2)
    gx, gy = np.meshgrid(small_grid.x, small_grid.y, indexing="ij")
    np.testing.assert_allclose(out[0], (1 + 2 * gx + 3 * gy + 4 * gx * gy * 1e3) * (1 + 0.5j), rtol=1e-10)
    assert np.all(out[1] == 0)


def test_binary_input_round_trip(tmp_path) -> None:
    x = np.linspace(0, 1e-2, 5)
    y = np.linspace(0, 2e-2, 4)
    h = np.arange(2 * 5 * 4).reshape(2, 5, 4) * (1 - 1j)
    np.savetxt(tmp_path / "x.txt", x)
    np.savetxt(tmp_path / "y.txt", y)
    real, imag = [], []
    for n in range(2):
        real.append(tmp_path / f"r{n}.bin")
        imag.append(tmp_path / f"i{n}.bin")
        h[n].real.ravel(order="F").tofile(real[-1])
        h[n].imag.ravel(order="F").tofile(imag[-1])
    opts = Options.from_dict(
        {
            "Initial": {
                "type": "binary",
                "InputFile": {
                    "xcord": "x.txt",
                    "ycord": "y.txt",
                    "real": [r.name for r in real],
                    "imag": [i.name for i in imag],
                },
            }
        },
        base_dir=tmp_path,
    )
    m = pw.WaveModel.from_options(opts)
    assert m.source is not None
    np.testing.assert_array_equal(m.source.harmonics, h)


def test_options_errors() -> None:
    with pytest.raises(TypeError, match="origin"):
        pw.WaveModel.from_options(Options({"Geometry": {"origin": [0, 0]}}))
    with pytest.raises(TypeError, match="lx"):
        pw.WaveModel.from_options(Options({"Geometry": {"lx": "big"}}))


def test_absorbing_layer_options() -> None:
    model = pw.WaveModel.from_options(Options({"AbsorbingBC": {"type": "PML", "width": 0.004, "reflection": 1e-3}}))
    layer = model.layer
    assert layer.kind == "pml" and layer.levels == 4 and layer.width == 0.004
    assert layer.strength == pytest.approx(3 * math.log(1e3) / (2 * 0.004))
    model = pw.WaveModel.from_options(Options({"AbsorbingBC": {"width": 0.002, "strength": 2000, "levels": 6}}))
    assert model.layer.kind == "sponge" and model.layer.strength == 2000 and model.layer.levels == 6
    with pytest.raises(ValueError):
        pw.WaveModel.from_options(Options({"AbsorbingBC": {"type": "cpml", "width": 0.002, "strength": 1.0}}))
