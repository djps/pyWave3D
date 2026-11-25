"""Run the reference options file ``options/woptions.json`` and its source ``data/pxy_at_z0.txt``.

The source in ``data/pxy_at_z0.txt`` is the first harmonic on a 50 x 50 grid over 10 mm: a
uniform, in-phase square piston of 9.6 mm at 1 MPa and 1 MHz. (The ``Model.Transducer`` section
of the file is not used: an ``Initial.InputFile`` takes precedence.)

Two runs:

1. ``reference``: the file exactly as given: linear (``"NLN": false``), lossy, a 10 x 10 x 20 mm
   domain, 4 harmonics. The piston fills the periodic transverse domain, so this is close to a
   plane wave with a little diffraction from the gaps between the images; the output points
   beyond z = 20 mm are skipped.
2. ``wide``: the same source data centred in a 60 x 60 mm domain, 100 mm long, nonlinear with
   16 harmonics, 6 points per wavelength and a PML along the edges, so the piston radiates as
   it would into open water (near field to about (D/2)^2 / lambda = 15 mm, then a diverging far
   field) and the output points at z = 50 and 100 mm are inside the domain. Linear runs of this
   case on this domain agree with a 120 mm wide one to 5% on the axis up to 100 mm; on 40 mm
   with the PML to 9%, and without it to 37%.

Each run writes the usual outputs (``axial.txt``, ``planes.txt``, ``metrics.json``,
``focal_plane.txt``, ``results.npz``, point waveforms) to ``<out>/<name>/``, and with matplotlib
installed a figure ``<out>/woptions.png``.

Usage (from the repository root)::

    python examples/woptions_example.py [--out woptions_out] [--workers 4] [--no-plot]

Install pywave3d first (``pip install -e .`` in the repository root). Without that, the script imports it from
``src/``, which still needs numpy, scipy and beartype.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

try:
    import pywave3d as pw
except ModuleNotFoundError:  # not installed: use the sources next to this example
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    try:
        import pywave3d as pw
    except ModuleNotFoundError as err:
        sys.exit(f"pywave3d needs '{err.name}': run `pip install -e .` in the repo root (or `pip install {err.name}`)")
from pywave3d.options import Options

REPO = Path(__file__).resolve().parents[1]
OPTIONS = REPO / "options" / "woptions.json"


def reference_options() -> Options:
    """``options/woptions.json`` as it is."""
    return Options.load(OPTIONS)


def wide_options() -> Options:
    """The same file and source, in a domain wide and long enough to see the beam."""
    base = Options.load(OPTIONS)
    data = base.to_dict()
    model, solver = data["Model"], data["Solver"]
    # the data cover 0 <= x, y <= 10 mm: centre them in 60 mm
    model["Geometry"].update({"lx": 0.06, "ly": 0.06, "lz": 0.10, "origin": [-0.025, -0.025, 0.0]})
    model["AbsorbingBC"] = {"type": "pml", "width": 0.004, "reflection": 1e-4, "power": 2}
    solver["Options"].update({"NLN": True, "AbsorbingBC": True})
    # 6 points per wavelength, 32 samples per period (16 harmonics)
    solver["DiffractSolver"].update({"NumberOfPointsPerWaveLength": 6, "SamplingFrequencyMultiplier": 32})
    solver["Metrics"]["levels_dB"] = [-6, -3]
    solver["Output"]["every"] = 2
    return Options.from_dict(data, base.base_dir)  # base_dir: pxy_at_z0.txt is found in ../data


def run(name: str, opts: Options, out: Path, workers: int) -> pw.Results:
    t = time.perf_counter()
    res = pw.simulate(opts, workers=workers)
    harmonics = opts.get("Solver.Metrics.harmonics", int, 4)
    res.save(out / name, harmonics=harmonics)
    m = res.metrics()
    f = m["focus"]
    print(f"\n{name}: {res.grid.summary()}")
    print(f"  {time.perf_counter() - t:.1f} s, outputs in {out / name}")
    print(
        f"  peak of p+ on the axis at z = {f['z'] * 1e3:.1f} mm: p+ = {f['p_positive'] / 1e6:.3f} MPa, "
        f"p- = {f['p_negative'] / 1e6:.3f} MPa, |P1..P4| = "
        + ", ".join(f"{h / 1e6:.3f}" for h in f["harmonics"][:4])
        + " MPa"
    )
    iec = m["iec"]
    power = iec["power"]["exact"]
    print(
        f"  power at the source {power['source']:.2f} W, at the end {power['end']:.2f} W; "
        f"I_spta {iec['I_spta'] / 1e4:.1f} W/cm^2; p_c/p_r = {iec['nonlinear']['p_c_over_p_r']:.3f}"
    )
    return res


def plot(results: dict[str, pw.Results], out: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib is not installed: no figure")
        return
    fig, ax = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for name, res in results.items():
        z = res.z * 1e3
        ax[0, 0].plot(z, np.abs(res.axis[:, 0]) / 1e6, label=f"{name}: |P1|")
        ax[0, 0].plot(z, res.axis_p_plus / 1e6, "--", label=f"{name}: p+")
    ax[0, 0].set(xlabel="z [mm]", ylabel="pressure on the axis [MPa]", title="Axis")
    ax[0, 0].legend(fontsize=8)

    wide = results["wide"]
    z = wide.z * 1e3
    for n in range(4):
        ax[0, 1].semilogy(z, np.abs(wide.axis[:, n]) / 1e6, label=f"|P{n + 1}|")
    ax[0, 1].set(xlabel="z [mm]", ylabel="[MPa]", title="wide: harmonics on the axis", ylim=(1e-4, None))
    ax[0, 1].legend(fontsize=8)

    g = wide.grid
    ext = (g.z0 * 1e3, z[-1], g.x[0] * 1e3, g.x[-1] * 1e3)
    ax[1, 0].imshow(wide.xz_amplitude.T / 1e6, origin="lower", aspect="auto", extent=ext, cmap="viridis")
    ax[1, 0].set(xlabel="z [mm]", ylabel="x [mm]", title="wide: |P1| in the xz plane [MPa]")

    for name, (pos, harmonics) in wide.points.items():
        tau, p = wide.waveform_at(harmonics)
        ax[1, 1].plot(tau * 1e6, p / 1e6, label=f"{name}: z = {pos[2] * 1e3:.0f} mm")
    ax[1, 1].set(xlabel="retarded time [us]", ylabel="p [MPa]", title="wide: waveforms at the output points")
    ax[1, 1].legend(fontsize=8)
    path = out / "woptions.png"
    fig.savefig(path, dpi=120)
    print(f"\nfigure: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("woptions_out"), help="output directory")
    parser.add_argument("--workers", type=int, default=4, help="threads")
    parser.add_argument("--no-plot", action="store_true", help="do not draw the figure")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    print(f"options: {OPTIONS}")
    results = {
        "reference": run("reference", reference_options(), args.out, args.workers),
        "wide": run("wide", wide_options(), args.out, args.workers),
    }
    summary = {name: res.metrics()["focus"] for name, res in results.items()}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    if not args.no_plot:
        plot(results, args.out)


if __name__ == "__main__":
    main()
