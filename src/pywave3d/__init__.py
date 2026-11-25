"""pyWave3D: nonlinear ultrasound beams by a pseudo-spectral spatial-marching scheme.

The pressure on a plane z = const is held as the complex amplitudes of its harmonics in the
retarded time tau = t - z/c0,

    p(x, y, z, tau) = Re sum_{n=1}^{K} P_n(x, y, z) exp(+i n omega tau),

and marched along z with second-order (Strang) operator splitting of the one-way Westervelt
(Khokhlov-Zabolotskaya-Kuznetsov type) equation:

* diffraction and absorption together, exactly, per harmonic in the angular spectrum
  (kx, ky) domain (:mod:`pywave3d.linear`);
* the quadratic nonlinearity, either in the frequency domain (dealiased pseudo-spectral
  Runge-Kutta) or, once shocks form, in the time domain with a conservative Godunov-type
  scheme (:mod:`pywave3d.nonlinear`).

This is the approach of the Moscow State University HIFU simulators (Yuldashev and
Khokhlova 2011; Yuldashev et al. 2021). The default model is water at 20 C with
thermoviscous absorption (alpha proportional to f^2) and nonlinearity switched on.

Every public function and method is checked at run time with beartype, unless the
environment variable ``PYWAVE3D_NO_BEARTYPE`` is set before the package is imported.
"""

import os

if not os.environ.get("PYWAVE3D_NO_BEARTYPE"):
    from beartype import BeartypeConf
    from beartype.claw import beartype_this_package

    # is_pep484_tower: an int is accepted where a float is expected, and a float where a complex is
    beartype_this_package(conf=BeartypeConf(is_pep484_tower=True))

from pywave3d.grid import Grid
from pywave3d.linear import AbsorbingLayer, LinearPropagator
from pywave3d.medium import Attenuation, Medium
from pywave3d.model import Source, Transducer, WaveModel
from pywave3d.nonlinear import NonlinearOperator
from pywave3d.options import Options, load_options
from pywave3d.recorder import PlaneRecorder, RecordSettings
from pywave3d.results import Results
from pywave3d.solver import SolverSettings, WaveSolver, simulate

__version__ = "0.2.0"

__all__ = [
    "AbsorbingLayer",
    "Attenuation",
    "Grid",
    "LinearPropagator",
    "Medium",
    "NonlinearOperator",
    "Options",
    "PlaneRecorder",
    "RecordSettings",
    "Results",
    "SolverSettings",
    "Source",
    "Transducer",
    "WaveModel",
    "WaveSolver",
    "__version__",
    "load_options",
    "simulate",
]
