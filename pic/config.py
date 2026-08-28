from dataclasses import dataclass
from enum import Enum


class BoundaryCondition(Enum):
    """Boundary choices retained by the original solver interface."""

    Open = 0
    Absorbing = 1
    Periodic = 2


@dataclass
class Parameters:
    """Numerical controls shared by the grid-based solver drivers."""

    x_max: float
    n_cells: int
    t_max: float
    max_iter: int
    bc: BoundaryCondition
    dimX: int
    dimV: int
    num_particles: int = -1
    damping_width: float = 0
    SOR_max_iter: int = 1000
    SOR_tol: float = 1.0e-6
    SOR_omega: float = 1.5
    seed: int = 42
    dx: float = 0.0
    n0: float = 1.0
    time_safety_factor: float = 20.0

    def __post_init__(self):
        if self.x_max <= 0:
            raise ValueError("x_max must be positive")
        if self.n_cells <= 0:
            raise ValueError("n_cells must be positive")
        if self.dimX <= 0 or self.dimV <= 0:
            raise ValueError("dimX and dimV must be positive")
        self.dx = self.x_max / self.n_cells
        if self.n0 <= 0:
            raise ValueError("n0 must be positive")
        if self.time_safety_factor <= 0:
            raise ValueError("time_safety_factor must be positive")

    def __repr__(self):
        field_strings = [f"{field}: {getattr(self, field)}" for field in self.__dataclass_fields__]
        return f"{self.__class__.__name__}(\n  " + ",\n  ".join(field_strings) + "\n)"

    __str__ = __repr__
