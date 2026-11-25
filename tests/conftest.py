"""Shared fixtures: small grids and models that run in well under a second."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import pywave3d as pw

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def repo() -> Path:
    """Root of the pyWave3D repository."""
    return REPO


@pytest.fixture
def water() -> pw.Medium:
    return pw.Medium.water()


@pytest.fixture
def small_grid() -> pw.Grid:
    """32 x 32 points of 0.25 mm, 21 planes of 0.25 mm, 8 samples per period."""
    return pw.Grid(nx=32, ny=32, nz=21, nt=8, dx=2.5e-4, dy=2.5e-4, dz=2.5e-4)


def focused_model(
    radius: float = 6e-3, depth: float = 30e-3, lx: float = 32e-3, lz: float = 45e-3, amplitude: float = 1e5
) -> pw.WaveModel:
    """A focused piston in water on a domain wide enough for the angular spectrum method."""
    return pw.WaveModel(
        lengths=(lx, lx, lz),
        transducer=pw.Transducer(outer_radius=radius, focusing_depth=depth, amplitude=amplitude),
    )


def rayleigh_axis(model: pw.WaveModel, grid: pw.Grid, z: np.ndarray) -> np.ndarray:
    """|p_1| on the beam axis from the Rayleigh-Sommerfeld integral over the sampled source."""
    k = 2 * np.pi * model.frequency / model.medium.sound_speed
    src = model.initial_field(grid)[0]
    xc, yc = grid.centre
    xx, yy = np.meshgrid(grid.x - xc, grid.y - yc, indexing="ij")
    m = np.abs(src) > 0
    ps, r2 = src[m], (xx**2 + yy**2)[m]
    out = np.empty(z.size)
    for i, zi in enumerate(z):
        if zi <= 0:
            out[i] = np.nan
            continue
        R = np.sqrt(r2 + zi**2)
        out[i] = (
            abs(np.sum(ps * (zi / R) * (1 / R - 1j * k) * np.exp(-1j * k * R) / R)) * grid.dx * grid.dy / (2 * np.pi)
        )
    return out
