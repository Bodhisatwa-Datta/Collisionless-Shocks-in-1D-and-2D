import sys

sys.path.append("./")
sys.path.append("../")

import numpy as np

from initial_distributions import two_stream
from parameters import BoundaryCondition, Parameters
import solver_1D3V


if __name__ == "__main__":
    np.random.seed(42)

    num_particles = 4_000
    num_cells = 100
    x_max = 1.0
    electrons, ions = two_stream(
        num_particles,
        x_max,
        v_the=0.01,
        v_bulk=0.0,
        nx=num_cells,
        eps=1e-3,
        mode=1,
        dimV=3,
    )
    params = Parameters(
        x_max=x_max,
        n_cells=num_cells,
        t_max=2.0,
        max_iter=5_000,
        bc=BoundaryCondition.Periodic,
        dimX=1,
        dimV=3,
        num_particles=num_particles,
        n0=1.0,
    )

    results = solver_1D3V.simulate(electrons, ions, params, write_results=False)
    print("maximum saved Gauss residual:", max(results.gauss_linf))
    print("maximum saved continuity residual:", max(results.continuity_linf))
    relative_energy_drift = max(
        abs((energy - results.TE[0]) / results.TE[0]) for energy in results.TE
    )
    print("maximum relative total-energy drift:", relative_energy_drift)
    print(
        "maximum transverse field energy:",
        max(results.electric_transverse_energy) + max(results.magnetic_energy),
    )
