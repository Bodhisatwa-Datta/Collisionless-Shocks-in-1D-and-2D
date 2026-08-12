from datetime import datetime
import time

import numpy as np

import boundary_conditions
from grids import Grid1D3V
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
    """Run the validated periodic 1D3V electromagnetic PIC cycle."""
    np.random.seed(params.seed)
    setup_start = time.time()

    if params.bc is not BoundaryCondition.Periodic:
        raise NotImplementedError(
            "The validated 1D3V electromagnetic solver currently supports "
            "periodic particle and field boundaries only"
        )

    grid = Grid1D3V(params.x_max, params.n_cells)
    results = ResultsND()

    # One-dimensional unit-area macro-particle weights. Changing particles per
    # cell now changes noise, not the continuum density or plasma frequency.
    electrons.weight = params.n0 * params.x_max / electrons.N
    ions.weight = params.n0 * params.x_max / ions.N

    max_v = max(np.max(np.abs(electrons.v)), np.max(np.abs(ions.v)))
    dt = calculate_dt_max(
        params.dx,
        max_v,
        electrons.qm,
        electrons.dimX,
        safety_factor=20,
        number_density=params.n0,
    )

    # At t=0, deposit charge, satisfy periodic Gauss law, then initialize
    # the particle velocity at the leapfrog half step.
    grid.set_densities(electrons, ions)
    maxwell.initialize_electric_field_1D3V(grid)
    newton.initialize_velocities_half_step_1D3V(grid, electrons, ions, params, dt)

    KE = electrons.relativistic_kinetic_energy() + ions.relativistic_kinetic_energy()
    PE = (grid.dx / 2) * (
        eps_0 * np.sum(grid.E**2) + np.sum(grid.B**2) / mu_0
    )
    TE = KE + PE
    results.save(0, KE, PE, TE, grid, electrons, ions)

    print(f"Setup phase took {time.time() - setup_start:.3f} seconds")
    print("iteration        time          dt  wall-clock time [s]  Total Energy")

    run_start = time.time()
    t_last = run_start
    t = 0.0
    step = 0
    save_interval = 50

    while t < params.t_max and step < params.max_iter:
        step += 1
        t += dt

        rho_old = grid.rho.copy()
        electron_x_old = electrons.x.copy()
        ion_x_old = ions.x.copy()

        # Drift x^n -> x^(n+1) with v^(n+1/2).
        newton.advance_positions(electrons, dt)
        newton.advance_positions(ions, dt)
        boundary_conditions.periodic_bc(electrons, ions, params.x_max)

        # Deposit rho^(n+1) and J^(n+1/2). The longitudinal current is
        # constructed to satisfy the periodic discrete continuity equation.
        grid.set_densities(electrons, ions)
        maxwell.calc_charge_conserving_current_1D3V(
            grid,
            electrons,
            ions,
            electron_x_old,
            ion_x_old,
            rho_old,
            dt,
        )

        # Advance E and B together, then monitor the preserved Gauss constraint.
        maxwell.calc_fields_1D3V(grid, dt)
        grid.gauss_residual[:] = maxwell.calc_gauss_residual_1D3V(grid)

        # Kick v^(n+1/2) -> v^(n+3/2) at x^(n+1).
        newton.boris_pusher_1D3V(grid, electrons, dt)
        newton.boris_pusher_1D3V(grid, ions, dt)

        if (step + 1) % save_interval == 0:
            KE_prev = (
                electrons.relativistic_kinetic_energy()
                + ions.relativistic_kinetic_energy()
            )

        if step % save_interval == 0:
            KE = (
                electrons.relativistic_kinetic_energy()
                + ions.relativistic_kinetic_energy()
                + KE_prev
            ) / 2
            PE = (grid.dx / 2) * (
                eps_0 * np.sum(grid.E**2) + np.sum(grid.B**2) / mu_0
            )
            TE = KE + PE
            results.save(t, KE, PE, TE, grid, electrons, ions)

        if time.time() - t_last > 5:
            t_last = time.time()
            print(
                f"{step:9}{t:12.4e}{dt:12.4e}"
                f"{t_last - run_start:21.3e}{TE:14.4e}"
            )

    print(f"{step:9}{t:12.4e}{dt:12.4e}{time.time() - run_start:21.3e}{TE:14.4e}")
    print("DONE!")
    if write_results:
        string_time = datetime.fromtimestamp(run_start).strftime("%Y-%m-%dT%Hh%Mm%Ss")
        results.write(f"Results/{string_time}", params)
        print(f"Results saved in Results/{string_time}/")
    return results
