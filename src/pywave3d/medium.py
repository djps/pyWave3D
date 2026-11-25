"""The propagation medium: sound speed, density, nonlinearity and the attenuation law.

The defaults describe water at 20 C (Hamilton and Blackstock, *Nonlinear Acoustics*, 1998):
c0 = 1482 m/s, rho0 = 998 kg/m^3, beta = 1 + B/2A = 3.5 and a sound diffusivity
delta = 4.33e-6 m^2/s, so that the thermoviscous absorption, quadratic in frequency, is

    alpha(f) = delta omega^2 / (2 c0^3)  =  0.0263 Np/m at 1 MHz  (0.228 dB/m).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from pywave3d._types import RealArray

AttenuationLaw = Literal["thermoviscous", "power_law"]


@dataclass(frozen=True, slots=True)
class Attenuation:
    """Frequency dependence of the absorption.

    Attributes:
        law: ``"thermoviscous"`` (the default): alpha = delta omega^2 / (2 c0^3), set by the
            medium's diffusivity; or ``"power_law"``: alpha_n = alpha0 n^power at the n-th
            harmonic of the base frequency.
        alpha0: power law only: attenuation at the base frequency [Np/m].
        power: power law only: exponent y (0 < y <= 2).
        dispersion: power law only: add the dispersion that the Kramers-Kronig relations
            require of a power law (Szabo 1994; Waters et al. 2005), so that the field stays
            causal. Phase speed equals c0 at the base frequency.
    """

    law: AttenuationLaw = "thermoviscous"
    alpha0: float = 0.0
    power: float = 2.0
    dispersion: bool = False

    def __post_init__(self) -> None:
        if self.law not in ("thermoviscous", "power_law"):
            raise ValueError(f"attenuation law must be 'thermoviscous' or 'power_law', not {self.law!r}")
        if self.alpha0 < 0:
            raise ValueError("alpha0 must be >= 0")
        if self.law == "power_law" and not 0.0 < self.power <= 3.0:
            raise ValueError("the power-law exponent must be in (0, 3]")


@dataclass(frozen=True, slots=True)
class Medium:
    """A homogeneous, lossy, quadratically nonlinear fluid.

    Attributes:
        sound_speed: small-signal sound speed c0 [m/s].
        density: ambient density rho0 [kg/m^3].
        nonlinearity: coefficient of nonlinearity beta = 1 + B/(2A).
        diffusivity: sound diffusivity delta [m^2/s] (thermoviscous law).
        attenuation: the absorption law.
    """

    sound_speed: float = 1482.0
    density: float = 998.0
    nonlinearity: float = 3.5
    diffusivity: float = 4.33e-6
    attenuation: Attenuation = field(default_factory=Attenuation)

    def __post_init__(self) -> None:
        if self.sound_speed <= 0 or self.density <= 0:
            raise ValueError("sound speed and density must be positive")
        if self.diffusivity < 0 or self.nonlinearity < 0:
            raise ValueError("diffusivity and nonlinearity must be >= 0")

    @classmethod
    def water(cls) -> Medium:
        """Water at 20 C (the defaults)."""
        return cls()

    @property
    def impedance(self) -> float:
        """Characteristic impedance rho0 c0 [kg/(m^2 s)]."""
        return self.density * self.sound_speed

    @property
    def epsilon(self) -> float:
        """Nonlinear coefficient beta / (rho0 c0^3) of dp/dz = epsilon p dp/dtau [1/Pa/m*s]."""
        return self.nonlinearity / (self.density * self.sound_speed**3)

    def shock_distance(self, frequency: float, amplitude: float) -> float:
        """Plane-wave shock formation distance x_sh = rho0 c0^3 / (beta omega p0) [m]."""
        if self.nonlinearity == 0 or amplitude == 0:
            return math.inf
        return 1.0 / (self.epsilon * 2 * math.pi * frequency * amplitude)

    def alpha(self, frequency: float, n: RealArray) -> RealArray:
        """Attenuation [Np/m] of the harmonics ``n`` (multiples of ``frequency``)."""
        n = np.asarray(n, dtype=float)
        att = self.attenuation
        if att.law == "power_law":
            return att.alpha0 * n**att.power
        omega = 2 * math.pi * frequency * n
        return self.diffusivity * omega**2 / (2 * self.sound_speed**3)

    def dispersion(self, n: RealArray) -> RealArray:
        """Extra wavenumber dk_n [rad/m] of harmonic n from Kramers-Kronig, zero at n = 1.

        For alpha_n = alpha0 n^y: dk_n = alpha0 tan(pi y / 2) (n^y - n), with the limit
        -(2/pi) alpha0 n ln n at y = 1. Zero for the thermoviscous law or y = 2.
        """
        n = np.asarray(n, dtype=float)
        att = self.attenuation
        if att.law != "power_law" or not att.dispersion or att.power == 2.0:
            return np.zeros_like(n)
        y = att.power
        if abs(y - 1.0) < 1e-9:
            return -(2 / math.pi) * att.alpha0 * n * np.log(n)
        return att.alpha0 * math.tan(math.pi * y / 2) * (n**y - n)
