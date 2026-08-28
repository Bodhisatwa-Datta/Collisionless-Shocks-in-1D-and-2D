"""Periodic 1D3V shock-tube analogue.

The validated 1D3V solver currently supports periodic boundaries.  This case
therefore places two neutral, oppositely directed plasma slabs in a periodic
box.  They collide at x=L/2 and move apart at the periodic seam.  This is the
clean periodic counterpart of the original moving-slab 1D1V experiment.
"""

import numpy as np

from pic import solver_1d3v
from pic.config import BoundaryCondition, Parameters
from pic.constants import m_e, q_e, q_i
from pic.particles import Particles


X_MAX = 20.0
N_CELLS = 128
PARTICLES_PER_SPECIES = 4_096
ION_MASS = 25.0
BULK_SPEED = 0.03
ELECTRON_THERMAL_SPEED = 0.10
ION_THERMAL_SPEED = 0.002
T_MAX = 50.0
SEED = 23


def quiet_shock_tube_1d3v(seed=SEED):
    """Return initially neutral, counter-streaming plasma slabs."""
    rng = np.random.default_rng(seed)
    count = PARTICLES_PER_SPECIES
    positions = ((np.arange(count) + 0.5) * X_MAX / count).reshape(-1, 1)

    electrons = Particles(count, 1, 3, m_e, q_e)
    ions = Particles(count, 1, 3, ION_MASS, q_i)
    electrons.x[:] = positions
    ions.x[:] = positions

    flow = np.where(positions[:, 0] < X_MAX / 2, BULK_SPEED, -BULK_SPEED)
    electrons.v.fill(0.0)
    ions.v.fill(0.0)
    electrons.v[:, 0] = rng.normal(0.0, ELECTRON_THERMAL_SPEED, count) + flow
    ions.v[:, 0] = rng.normal(0.0, ION_THERMAL_SPEED, count) + flow

    # vy=vz=0 and E_perp=B=0 make this the controlled longitudinal embedding
    # of the 1D1V experiment in the repaired 1D3V algorithm.
    electrons.v_to_u()
    ions.v_to_u()
    return electrons, ions


def run(write_results=False):
    electrons, ions = quiet_shock_tube_1d3v()
    params = Parameters(
        x_max=X_MAX,
        n_cells=N_CELLS,
        t_max=T_MAX,
        max_iter=30_000,
        bc=BoundaryCondition.Periodic,
        dimX=1,
        dimV=3,
        num_particles=2 * PARTICLES_PER_SPECIES,
        seed=SEED,
        n0=1.0,
        time_safety_factor=40.0,
    )
    return solver_1d3v.simulate(
        electrons, ions, params, write_results=write_results
    )


if __name__ == "__main__":
    result = run(write_results=True)
    print("maximum saved Gauss residual:", max(result.gauss_linf))
    print("maximum saved continuity residual:", max(result.continuity_linf))
    print(
        "maximum relative total-energy drift:",
        max(abs((energy - result.TE[0]) / result.TE[0]) for energy in result.TE),
    )
