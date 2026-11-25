"""The example scripts: their options build and the quick runs work."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

import pywave3d as pw

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_example_options_file_is_valid() -> None:
    opts = pw.load_options(EXAMPLES / "water_focused.json")
    model = pw.WaveModel.from_options(opts.child("Model"))
    assert model.make_grid(opts.child("Solver.DiffractSolver")).nx > 0


def test_woptions_example(tmp_path: Path) -> None:
    ex = load("woptions_example")
    res = ex.run("reference", ex.reference_options(), tmp_path, 1)
    assert (tmp_path / "reference" / "axial.txt").is_file()
    assert json.loads((tmp_path / "reference" / "metrics.json").read_text())["focus"]["p_positive"] > 1e6
    assert res.grid.nx == 56 and not np.any(res.axis[:, 1])  # linear: no harmonics generated

    wide = ex.wide_options()
    model = pw.WaveModel.from_options(wide.child("Model"))
    grid = model.make_grid(wide.child("Solver.DiffractSolver"))
    settings = pw.SolverSettings.from_options(wide.child("Solver"))
    assert model.layer.kind == "pml" and settings.absorbing_layer and settings.nonlinearity
    assert grid.n_harmonics == 16 and grid.lengths[0] == pytest.approx(0.06)
    # the source data (0 <= x, y <= 10 mm) sit in the middle of the wider domain
    src = np.abs(model.initial_field(grid)[0])
    ix, iy = grid.centre_index
    assert src[ix, iy] == pytest.approx(1e6) and src[0, 0] == 0
