"""Public interface for the collisionless-shock PIC code.

The original project exposed a collection of modules at repository level. New
notebooks and experiments can import the common data types from :mod:`pic`; the
old module paths remain available so archived scripts continue to run.
"""

from .config import BoundaryCondition, Parameters
from .diagnostics import ShockMetrics, analyze_shock
from .grids import Grid1D, Grid1D3V, Grid2D
from .particles import Particles

__all__ = [
    "BoundaryCondition",
    "Grid1D",
    "Grid1D3V",
    "Grid2D",
    "Parameters",
    "Particles",
    "ShockMetrics",
    "analyze_shock",
]
