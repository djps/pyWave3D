"""Turn the output of other acoustic solvers into :class:`~pywave3d.results.Results`.

Every field parameter of :class:`~pywave3d.results.Results` (see ``docs/iec_outputs.md``) is
computed from transverse planes holding the harmonics of a periodic field. Each module here
reduces one kind of solver output to such planes and passes them, in increasing z, to a
:class:`~pywave3d.recorder.PlaneRecorder`:

* :mod:`~pywave3d.preprocess.spectral`: frequency-domain pseudo-spectral (marching,
  angular-spectrum, KZK) solvers, which already give harmonics on planes;
* :mod:`~pywave3d.preprocess.time_domain`: time-domain pseudo-spectral (k-space) solvers such as
  k-wave-python, whose sensor time series are fitted for harmonics over a steady window;
* :mod:`~pywave3d.preprocess.fdtd`: finite-difference time-domain solvers such as fullwave25,
  as the time-domain case with the drift of the absorbing layers removed and the sensor indices
  of fullwave25.

The result is the same :class:`~pywave3d.results.Results` as a pyWave3D run, with ``metrics()``
(including the IEC parameters) and ``save()``. A 2D field (one transverse axis) gives a grid with
``ny = 1`` and ``dy = 1 m``: its power and areas are per metre in y and its y widths are NaN.
"""

from pywave3d.preprocess import fdtd, spectral, time_domain
from pywave3d.preprocess._common import HarmonicFit

__all__ = ["HarmonicFit", "fdtd", "spectral", "time_domain"]
