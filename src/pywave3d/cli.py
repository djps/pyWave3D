"""Command line: ``pywave3d [nthreads] [options.json] [-o DIR] [-q | -v] [--no-progress]
[--precision {single,double}]
[--diffraction {full,parabolic,pade}] [--pade-order {1,2,3}]``.

Without an options file the built-in default runs: a 50 W, 1 MHz focused transducer
(radius 12.5 mm, focus 50 mm) in water, with nonlinearity and thermoviscous absorption.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from pywave3d.linear import PADE_ORDERS
from pywave3d.options import Options
from pywave3d.solver import DIFFRACTION_TYPES, PRECISIONS, simulate


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser."""
    p = argparse.ArgumentParser(prog="pywave3d", description=__doc__.splitlines()[0])
    p.add_argument("nthreads", type=int, nargs="?", default=1, help="threads for the FFTs (default 1)")
    p.add_argument(
        "options", type=Path, nargs="?", default=None, help="JSON options file (default: built-in water example)"
    )
    p.add_argument("-o", "--output", type=Path, default=Path("."), help="directory for the output files (default: .)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("-q", "--quiet", action="store_true", help="warnings and errors only")
    g.add_argument("-v", "--verbose", action="store_true", help="debug output")
    p.add_argument(
        "--no-progress",
        dest="progress_bar",
        action="store_false",
        default=None,
        help="no progress bar over the march (overrides Solver.Output.progress_bar)",
    )
    p.add_argument(
        "--precision",
        choices=PRECISIONS,
        default=None,
        help="precision of the march: complex128 (double) or complex64 (single) fields "
        "(overrides Solver.precision; default double)",
    )
    p.add_argument(
        "--diffraction",
        choices=DIFFRACTION_TYPES,
        default=None,
        help="diffraction operator: exact angular spectrum (full), KZK (parabolic) or a Pade approximant of the "
        "exact one (pade) (overrides Solver.DiffractSolver.Type; default full)",
    )
    p.add_argument(
        "--pade-order",
        type=int,
        choices=PADE_ORDERS,
        default=None,
        help="order m of the [m/m] Pade operator; implies --diffraction pade (overrides "
        "Solver.DiffractSolver.PadeOrder; default 1)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Run the simulation; return the exit status."""
    args = build_parser().parse_args(argv)
    level = logging.WARNING if args.quiet else logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s", stream=sys.stdout)
    try:
        opts = Options.load(args.options) if args.options is not None else Options({})
        results = simulate(
            opts,
            workers=max(1, args.nthreads),
            progress_bar=args.progress_bar,
            precision=args.precision,
            diffraction=args.diffraction,
            pade_order=args.pade_order,
        )
    except (OSError, TypeError, ValueError) as err:
        print(f"pywave3d: error: {err}", file=sys.stderr)
        return 1
    met = opts.child("Solver.Metrics")
    files = results.save(args.output, harmonics=met.get("harmonics", int, 4))
    m = results.metrics()
    logging.getLogger("pywave3d").info(
        "Focus at z = %.4e m: p+ = %.4e Pa, p- = %.4e Pa; wrote %s",
        m["focus"]["z"],
        m["focus"]["p_positive"],
        m["focus"]["p_negative"],
        ", ".join(f.name for f in files),
    )
    logging.getLogger("pywave3d").debug("%s", json.dumps(m, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
