import numpy as np

from pic import solver_2d2v
from pic.config import BoundaryCondition, Parameters
from pic.constants import m_e, m_i, q_e, q_i
from pic.particles import Particles


def quiet_periodic_plasma(particles_per_side, x_max, thermal_speed, seed=42):
    rng = np.random.default_rng(seed)
    coordinates = (np.arange(particles_per_side) + 0.5) * (
        x_max / particles_per_side
    )
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="ij")
    positions = np.column_stack((xx.ravel(), yy.ravel()))
    particle_count = len(positions)

    electrons = Particles(particle_count, 2, 2, m_e, q_e)
    ions = Particles(particle_count, 2, 2, m_i, q_i)
    electrons.x[:] = positions
    ions.x[:] = positions
    electrons.v[:] = rng.normal(0.0, thermal_speed, (particle_count, 2))
    # Remove finite-sample drift so no uniform current is initialized.
    electrons.v -= np.mean(electrons.v, axis=0, keepdims=True)
    ions.v.fill(0.0)
    electrons.v_to_u()
    ions.v_to_u()
    return electrons, ions


if __name__ == "__main__":
    electrons, ions = quiet_periodic_plasma(
        particles_per_side=32,
        x_max=1.0,
        thermal_speed=0.04,
    )
    params = Parameters(
        x_max=1.0,
        n_cells=24,
        t_max=0.25,
        max_iter=2_000,
        bc=BoundaryCondition.Periodic,
        dimX=2,
        dimV=2,
        num_particles=electrons.N + ions.N,
        n0=1.0,
    )
    results = solver_2d2v.simulate(
        electrons, ions, params, write_results=False
    )
    relative_energy_drift = max(
        abs((energy - results.TE[0]) / results.TE[0]) for energy in results.TE
    )
    print("maximum saved Gauss residual:", max(results.gauss_linf))
    print("maximum saved continuity residual:", max(results.continuity_linf))
    print("maximum relative total-energy drift:", relative_energy_drift)
