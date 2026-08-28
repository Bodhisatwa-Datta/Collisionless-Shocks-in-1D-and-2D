from math import sqrt
from .constants import c


def calculate_dt_max(dx, v, qm_e, dimx, safety_factor=5, number_density=1.0):
    """Return a conservative step from particle-crossing and plasma limits.

    ``qm_e`` is the electron charge-to-mass ratio in the active normalization.
    The electromagnetic crossing speed is bounded below by ``c`` and adjusted
    for the number of spatial dimensions.
    """

    # CFL condition (particle shouldn't cross more than one cell per timestep)
    dt_cfl = dx / (max(v, c) * sqrt(dimx))

    # Plasma frequency condition
    wp = sqrt(number_density * abs(qm_e))
    dt_wp = 2 / wp

    # Return the more restrictive timestep divided by the safety factor
    return min(dt_cfl, dt_wp) / safety_factor
