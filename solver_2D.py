from datetime import datetime
import time

import numpy as np

import boundary_conditions
from grids import Grid2D
import maxwell
import newton
from parameters import BoundaryCondition, Parameters
from particles import Particles
from physical_constants import eps_0, mu_0
from results import ResultsND
from time_constraint import calculate_dt_max


def simulate(
    electrons: Particles,
    ions: Particles,
    params: Parameters,
    write_results: bool = True,
):
    """Run the validated periodic 2D2V TMz electromagnetic PIC cycle."""
    if params.dimX != 2 or params.dimV != 2:
        raise ValueError("solver_2D requires dimX=2 and dimV=2")
    if params.bc is not BoundaryCondition.Periodic:
        raise NotImplementedError(
            "The validated 2D2V solver currently supports periodic boundaries only"
        )

    np.random.seed(params.seed)
    setup_start = time.time()
    grid = Grid2D(params.x_max, params.n_cells)
    results = ResultsND()

    domain_area = params.x_max * params.x_max
    electrons.weight = params.n0 * domain_area / electrons.N
    ions.weight = params.n0 * domain_area / ions.N
    max_v = max(np.max(np.abs(electrons.v)), np.max(np.abs(ions.v)))
    dt = calculate_dt_max(
        grid.dx,
        max_v,
        electrons.qm,
        2,
        safety_factor=20,
        number_density=params.n0,
    )

    grid.set_densities(electrons, ions)
    maxwell.initialize_electric_field_2D(grid)
    newton.boris_pusher_2D2V(grid, electrons, dt / 2)
    newton.boris_pusher_2D2V(grid, ions, dt / 2)

    kinetic = (
        electrons.relativistic_kinetic_energy()
        + ions.relativistic_kinetic_energy()
    )
    field = 0.5 * grid.dx * grid.dx * (
        eps_0 * np.sum(grid.E**2) + np.sum(grid.B**2) / mu_0
    )
    total = kinetic + field
    results.save(0.0, kinetic, field, total, grid, electrons, ions)

    print(f"Setup phase took {time.time() - setup_start:.3f} seconds")
    print("iteration        time          dt  wall-clock time [s]  Total Energy")
    run_start = time.time()
    last_log = run_start
    t = 0.0
    step = 0
    save_interval = 50

    while t < params.t_max and step < params.max_iter:
        step += 1
        t += dt
        rho_old = grid.rho.copy()
        electron_x_old = electrons.x.copy()
        ion_x_old = ions.x.copy()

        newton.advance_positions(electrons, dt)
        newton.advance_positions(ions, dt)
        boundary_conditions.periodic_bc(electrons, ions, params.x_max)
        grid.set_densities(electrons, ions)
        maxwell.calc_charge_conserving_current_2D(
            grid,
            electrons,
            ions,
            electron_x_old,
            ion_x_old,
            rho_old,
            dt,
        )
        maxwell.calc_fields_2D(grid, dt)
        grid.gauss_residual[:] = maxwell.calc_gauss_residual_2D(grid)
        newton.boris_pusher_2D2V(grid, electrons, dt)
        newton.boris_pusher_2D2V(grid, ions, dt)

        if step % save_interval == 0:
            kinetic = (
                electrons.relativistic_kinetic_energy()
                + ions.relativistic_kinetic_energy()
            )
            field = 0.5 * grid.dx * grid.dx * (
                eps_0 * np.sum(grid.E**2) + np.sum(grid.B**2) / mu_0
            )
            total = kinetic + field
            results.save(t, kinetic, field, total, grid, electrons, ions)

        if time.time() - last_log > 5:
            last_log = time.time()
            print(
                f"{step:9}{t:12.4e}{dt:12.4e}"
                f"{last_log - run_start:21.3e}{total:14.4e}"
            )

    print(f"{step:9}{t:12.4e}{dt:12.4e}{time.time()-run_start:21.3e}{total:14.4e}")
    print("DONE!")
    if write_results:
        timestamp = datetime.fromtimestamp(run_start).strftime("%Y-%m-%dT%Hh%Mm%Ss")
        results.write(f"Results/{timestamp}", params)
        print(f"Results saved in Results/{timestamp}/")
    return results
