"""Array type aliases used in the annotations (checked by beartype)."""

import numpy as np
import numpy.typing as npt

#: real array of any floating precision
RealArray = npt.NDArray[np.floating]
#: complex array of any precision
ComplexArray = npt.NDArray[np.complexfloating]
#: integer array
IntArray = npt.NDArray[np.integer]
#: boolean array
BoolArray = npt.NDArray[np.bool_]
