# Contains Maxwell solver / Poisson solver
# Solving Poisson's equation:
#   - https://en.wikipedia.org/wiki/Discrete_Poisson_equation
#   - https://en.wikipedia.org/wiki/Relaxation_(iterative_method)
#   - https://en.wikipedia.org/wiki/Successive_over-relaxation
import numpy as np

from .config import BoundaryCondition, Parameters
from .constants import c, eps_0
from .grids import Grid1D, Grid1D3V, Grid2D
from .particles import Particles


def poisson_solver(grid: Grid1D, electrons: Particles, ions: Particles, params: Parameters, tridiag, first=False):
    grid.set_densities(electrons, ions)

    # The legacy 1D1V path uses direct integration. The SOR implementation
    # below is retained for comparison but is not the default because its
    # convergence is too slow for a field solve at every particle step.
    # if first:
    #     naive_poisson_solver(grid, params.dx)
    # solve_poisson_sor(grid.phi, -grid.rho / eps_0, params.dx, params.bc, params.SOR_max_iter, params.SOR_tol, params.SOR_omega)
    naive_poisson_solver(grid, grid.dx)
    # thomas_solver(grid, params.dx, tridiag)

    # Electric field calculation
    # E_i = - (phi_i+1 - phi_i) / dx
    grid.E[:-1] = -(grid.phi[1:] - grid.phi[:-1]) / params.dx
    # take boundary conditions into account
    if params.bc is BoundaryCondition.Periodic:
        # Warning: you will get an (unphysically) large field at the right
        # boundary if phi(x) is not periodic
        grid.E[-1] = -(grid.phi[0] - grid.phi[-1]) / params.dx
    else:  # use second order interpolation to get the last value
        grid.E[-1] = 2 * grid.E[-3] - grid.E[-2]


def naive_poisson_solver(grid: Grid1D, dx: float):
    """
    ∇∙E = ρ(x)/ε & E(x) = -∇ɸ(x) ⇒ ∆ɸ(x) = -ρ(x)/ε ⇒ d^2ɸ(x)/dx^2 = -ρ(x)/ε ⇒
    ɸ(x) = -1/ε * ∬ρ(x) + C_1*x + C_2
    """
    # The additive potential constant is set to zero. Boundary-dependent
    # integration constants are not reconstructed in this legacy routine.
    grid.phi.fill(0)
    grid.phi[1:] = -1 * np.cumsum(np.cumsum(grid.rho[:-1])) * dx**2 / eps_0


def thomas_solver(grid: Grid1D, dx: float, tridiag):
    from scipy.sparse import linalg

    dens_phi = linalg.spsolve(tridiag, -dx * dx * grid.rho[1:] / eps_0)
    grid.phi = np.concatenate(([0], dens_phi))


def solve_poisson_sor(u, f, dx, bound_cond, max_iter=100000, tol=1e-4, omega=1.5):
    """
    Solve the Poisson equation: ∆u(x) = f(x)

    The 1D Gauss equation (∇∙E = ρ(x)/ε & E(x) = -∇ɸ(x)) has f(x) = -ρ(x)/ε

    Uses finite differences and solves the resulting linear system using Successive Over-Relaxation (SOR)
    """
    num_cells = np.size(f)

    if bound_cond is BoundaryCondition.Open or bound_cond is BoundaryCondition.Absorbing:
        for _ in range(max_iter):
            max_diff = 0
            for i in range(1, num_cells - 1):  # Inner grid points
                old_u = u[i]
                u[i] = (1 - omega) * u[i] + (omega / 2) * (u[i + 1] + u[i - 1] - dx**2 * f[i])
                max_diff = max(max_diff, abs(u[i] - old_u))

            # The legacy open/absorbing path closes the system by quadratic
            # extrapolation of the potential at each boundary.
            u[0] = 3 * u[1] - 3 * u[2] + u[3]
            u[-1] = 3 * u[-2] - 3 * u[-3] + u[-4]
            if max_diff < tol:
                break
    else:  # Periodic boundaries
        for _ in range(max_iter):
            max_diff = 0
            for i in range(num_cells - 1):  # Inner grid points
                old_u = u[i]
                u[i] = (1 - omega) * u[i] + (omega / 2) * (u[i + 1] + u[i - 1] - dx**2 * f[i])
                max_diff = max(max_diff, abs(u[i] - old_u))
            u[-1] = (1 - omega) * u[-1] + (omega / 2) * (u[0] + u[-2] - dx**2 * f[-1])
            if max_diff < tol:
                break


def calc_curr_dens_1D3V(grid: Grid1D3V, electrons: Particles, ions: Particles):
    """Deposit an instantaneous CIC current density at particle positions."""
    grid.J.fill(0)
    for particles in (electrons, ions):
        current = particles.q * particles.weight / grid.dx * particles.v
        np.add.at(
            grid.J,
            particles.idx.flatten(),
            current * (1 - particles.cic_weights),
        )
        np.add.at(
            grid.J,
            (particles.idx.flatten() + 1) % grid.n_cells,
            current * particles.cic_weights,
        )


def initialize_electric_field_1D3V(grid: Grid1D3V):
    """Initialize periodic Ex so D- Ex = rho/eps_0 to roundoff."""
    charge_scale = grid.dx * np.sum(np.abs(grid.rho))
    net_charge = grid.dx * np.sum(grid.rho)
    tolerance = 1e-12 * max(1.0, charge_scale)
    if abs(net_charge) > tolerance:
        raise ValueError(
            "A periodic 1D3V domain must be charge neutral; "
            f"integral(rho)={net_charge:.6e}"
        )

    grid.E[:, 0].fill(0)
    grid.E[1:, 0] = grid.dx * np.cumsum(grid.rho[1:]) / eps_0
    # The periodic Gauss solve determines Ex only up to a uniform field.
    grid.E[:, 0] -= np.mean(grid.E[:, 0])
    grid.gauss_residual[:] = calc_gauss_residual_1D3V(grid)


def calc_gauss_residual_1D3V(grid: Grid1D3V):
    """Return D- Ex - rho/eps_0 using the solver's periodic derivative."""
    return (grid.E[:, 0] - np.roll(grid.E[:, 0], 1)) / grid.dx - grid.rho / eps_0


def calc_charge_conserving_current_1D3V(
    grid: Grid1D3V,
    electrons: Particles,
    ions: Particles,
    electron_x_old,
    ion_x_old,
    rho_old,
    dt,
):
    """Deposit J^(n+1/2) and enforce periodic discrete continuity.

    Transverse current is CIC-deposited at each trajectory midpoint. In one
    periodic dimension, continuity determines longitudinal current up to its
    spatial mean; the mean is set from the particle trajectory flux.
    """
    grid.J.fill(0)
    mean_jx = 0.0

    for particles, x_old in ((electrons, electron_x_old), (ions, ion_x_old)):
        old = x_old[:, 0]
        new = particles.x[:, 0]
        displacement = (new - old + 0.5 * grid.x_max) % grid.x_max - 0.5 * grid.x_max
        midpoint = (old + 0.5 * displacement) % grid.x_max

        scaled = midpoint / grid.dx
        idx = np.floor(scaled).astype(np.int32)
        weight = scaled - idx
        transverse = particles.q * particles.weight / grid.dx * particles.v[:, 1:]
        np.add.at(grid.J[:, 1:], idx, transverse * (1 - weight[:, np.newaxis]))
        np.add.at(
            grid.J[:, 1:],
            (idx + 1) % grid.n_cells,
            transverse * weight[:, np.newaxis],
        )
        mean_jx += particles.q * particles.weight * np.sum(displacement) / (
            dt * grid.x_max
        )

    delta_rho = grid.rho - rho_old
    grid.J[0, 0] = 0.0
    grid.J[1:, 0] = -grid.dx / dt * np.cumsum(delta_rho[1:])
    grid.J[:, 0] += mean_jx - np.mean(grid.J[:, 0])

    grid.continuity_residual[:] = (
        delta_rho / dt
        + (grid.J[:, 0] - np.roll(grid.J[:, 0], 1)) / grid.dx
    )


def calc_fields_1D3V(grid: Grid1D3V, dt):
    """
    Solve Maxwell's equations on a spatial Yee grid with periodic boundaries.

    E and B are stored at the same integer time after this function returns.
    A time-centered B/2 -> E -> B/2 (velocity-Verlet) composition is
    used so that the source-free Maxwell update is stable for c*dt/dx <= 1.

    Maxwell's equations for 1D3V become:

    * dE_x/dt = -J_x / eps_0\n
    * dE_y/dt = -J_y / eps_0 - c² dB_z/dx\n
    * dE_z/dt = -J_z / eps_0 + c² dB_y/dx\n
    * dB_x/dt = 0\n
    * dB_y/dt = dE_z/dx\n
    * dB_z/dt = -dE_y/dx\n
    """
    half_dt = 0.5 * dt

    # Faraday: B^n -> B^(n+1/2), using E^n.
    grid.B[:, 1] += half_dt / grid.dx * (np.roll(grid.E[:, 2], -1) - grid.E[:, 2])
    grid.B[:, 2] -= half_dt / grid.dx * (np.roll(grid.E[:, 1], -1) - grid.E[:, 1])

    # Ampere-Maxwell: E^n -> E^(n+1), using B^(n+1/2) and J^(n+1/2).
    grid.E[:, 0] -= dt * grid.J[:, 0] / eps_0
    grid.E[:, 1] += dt * (
        -grid.J[:, 1] / eps_0
        - c * c / grid.dx * (grid.B[:, 2] - np.roll(grid.B[:, 2], 1))
    )
    grid.E[:, 2] += dt * (
        -grid.J[:, 2] / eps_0
        + c * c / grid.dx * (grid.B[:, 1] - np.roll(grid.B[:, 1], 1))
    )

    # Faraday: B^(n+1/2) -> B^(n+1), using E^(n+1).
    grid.B[:, 1] += half_dt / grid.dx * (np.roll(grid.E[:, 2], -1) - grid.E[:, 2])
    grid.B[:, 2] -= half_dt / grid.dx * (np.roll(grid.E[:, 1], -1) - grid.E[:, 1])


def calc_fields_1D3V_nonperiodic(grid: Grid1D3V, dt):
    E = grid.E.copy()
    max_change = 0.1  # Maximum allowed relative change
    # Calculate the fields at the full timestep
    grid.E[:, 0] += dt * -grid.J[:, 0] / eps_0
    grid.E[1:, 1] += dt * (-grid.J[1:, 1] / eps_0 - c * c / grid.dx * (grid.B[1:, 2] - grid.B[:-1, 2]))
    grid.E[1:, 2] += dt * (-grid.J[1:, 2] / eps_0 + c * c / grid.dx * (grid.B[1:, 1] - grid.B[:-1, 1]))
    dBy = dt / grid.dx * (E[1:, 2] - E[:-1, 2])
    dBz = -dt / grid.dx * (E[1:, 1] - E[:-1, 1])

    # Apply limiters to interior points
    dBy = np.clip(dBy, -max_change * np.abs(grid.B[:-1, 1]), max_change * np.abs(grid.B[:-1, 1]))
    dBz = np.clip(dBz, -max_change * np.abs(grid.B[:-1, 2]), max_change * np.abs(grid.B[:-1, 2]))

    grid.B[:-1, 1] += dBy
    grid.B[:-1, 2] += dBz

    # Calculate the boundary values using interpolation
    grid.E[0, 1] = 3 * grid.E[1, 1] - 3 * grid.E[2, 1] + grid.E[3, 1]
    grid.E[0, 2] = 3 * grid.E[1, 2] - 3 * grid.E[2, 2] + grid.E[3, 2]
    By_right = 3 * grid.B[-2, 1] - 3 * grid.B[-3, 1] + grid.B[-4, 1]
    Bz_right = 3 * grid.B[-2, 2] - 3 * grid.B[-3, 2] + grid.B[-4, 2]

    # Apply limiters to boundary values
    dBy_right = By_right - grid.B[-1, 1]
    dBz_right = Bz_right - grid.B[0, 2]

    dBy_right = np.clip(dBy_right, -max_change * np.abs(grid.B[-1, 1]), max_change * np.abs(grid.B[-1, 1]))
    dBz_right = np.clip(dBz_right, -max_change * np.abs(grid.B[-1, 2]), max_change * np.abs(grid.B[-1, 2]))

    grid.B[-1, 1] += dBy_right
    grid.B[0, 2] += dBz_right


def calc_E_1D3V(grid: Grid1D3V, dt, bc):
    """
    dE_x/dt = -J_x / eps_0\n
    dE_y/dt = -J_y / eps_0 - c² dB_z/dx\n
    dE_z/dt = -J_z / eps_0 + c² dB_y/dx\n
    """
    # The fields E and B are assumed to be known at the same gridpoints, we use upwind or downwind depending on the sign of the spatial derivative
    if bc is BoundaryCondition.Periodic:
        # calculate the fields at the full timestep
        # np.roll(Ez, -1) = [Ez(x1), Ez(x2), ..., Ez(xN), Ez(x0)]
        grid.E[:, 0] += dt * -grid.J[:, 0] / eps_0
        grid.E[:, 1] += dt * (-grid.J[:, 1] / eps_0 + c * c / grid.dx * (np.roll(grid.B[:, 2], 1) - grid.B[:, 2]))
        grid.E[:, 2] += dt * (-grid.J[:, 2] / eps_0 + c * c / grid.dx * (np.roll(grid.B[:, 1], -1) - grid.B[:, 1]))
    else:
        # calculate the fields at the full timestep
        grid.E[:, 0] += dt * -grid.J[:, 0] / eps_0
        grid.E[1:, 1] += dt * (-grid.J[1:, 1] / eps_0 + c * c / grid.dx * (grid.B[:-1, 2] - grid.B[1:, 2]))
        grid.E[:-1, 2] += dt * (-grid.J[:-1, 2] / eps_0 + c * c / grid.dx * (grid.B[1:, 1] - grid.B[:-1, 1]))

        # calculate the boundary values using interpolation
        grid.E[0, 1] = 3 * grid.E[1, 1] - 3 * grid.E[2, 1] + grid.E[3, 1]
        grid.E[-1, 2] = 3 * grid.E[-2, 2] - 3 * grid.E[-3, 2] + grid.E[-4, 2]


def calc_B_1D3V(grid: Grid1D3V, dt, bc):
    """
    dB_x/dt = 0\n
    dB_y/dt = dE_z/dx\n
    dB_z/dt = -dE_y/dx\n
    """
    if bc is BoundaryCondition.Periodic:
        # calculate the fields at the full timestep
        # np.roll(Ez, -1) = [Ez(x1), Ez(x2), ..., Ez(xN), Ez(x0)]
        grid.B[:, 1] += dt / grid.dx * (np.roll(grid.E[:, 2], -1) - grid.E[:, 2])
        grid.B[:, 2] += dt / grid.dx * (np.roll(grid.E[:, 1], 1) - grid.E[:, 1])
    else:
        # calculate the fields at the full timestep
        grid.B[:-1, 1] += dt / grid.dx * (grid.E[1:, 2] - grid.E[:-1, 2])
        grid.B[1:, 2] += dt / grid.dx * (grid.E[:-1, 1] - grid.E[1:, 1])
        # calculate the boundary values using interpolation
        grid.B[-1, 1] = 3 * grid.B[-2, 1] - 3 * grid.B[-3, 1] + grid.B[-4, 1]
        grid.B[0, 2] = 3 * grid.B[1, 2] - 3 * grid.B[2, 2] + grid.B[3, 2]


# def euler_solver_1D3V(grid: Grid1D3V, dt: float, bc: BoundaryCondition):
#     if bc is BoundaryCondition.Periodic:
#        # Solve via FFT
#        # Remove mean field, physically required for periodic boundary conditions as quasi neutrality must be maintained
#        grid.rho -= np.mean(grid.rho)
#        rho_k = np.fft.fft(grid.rho)
#        k = 2 * np.pi * np.fft.fftfreq(grid.n_cells) / grid.x_max  # k = 2 * pi * n / L (n = -N/2, -N/2+1, ..., N/2-1)
#
#        # Avoid dividing by k = zero
#        E_k = np.zeros_like(rho_k)
#        nonzero = k != 0
#        E_k[nonzero] = rho_k[nonzero] / (1j * k[nonzero] * eps_0)
#        grid.E[:, 0] = np.fft.ifft(E_k)
#    elif bc is BoundaryCondition.Absorbing:
#        # Solve using first order implicit discretization assuming E(x_0) = 0 (excluding any external fields)
#        grid.E[0, 0] = 0
#        for i in range(1, grid.n_cells):
#            grid.E[i, 0] = grid[i - 1, 0] + grid.dx / eps_0 * grid.rho[i]
#        grid.E[:, 0] += grid.E_0[:, 0]
#    elif bc is BoundaryCondition.Open:
#        # Solve using first order implicit discretization
#        grid.E[0, 0] = 0
#        for i in range(1, grid.n_cells):
#            grid.E[i, 0] = grid[i - 1, 0] + grid.dx / eps_0 * grid.rho[i]
#        # To remove any effects by setting E(x_0) = 0 we redo the calcs for E in opposite direction using E(x_N) as our starting point
#        # --> Not certain if this implementation is fully correct
#        for i in range(grid.n_cells - 2, -1, -1):
#            grid.E[i, 0] = grid[i + 1, 0] - grid.dx / eps_0 * grid.rho[i]
#        grid.E[:, 0] += grid.E_0[:, 0]


def calc_curr_dens_2D(grid: Grid1D3V, electrons: Particles, ions: Particles):
    raise RuntimeError(
        "Legacy 2D current deposition is disabled; use "
        "calc_charge_conserving_current_2D"
    )
    # Current density via CIC for both velocity components
    grid.J.fill(0)
    # Current projection here is defined for the periodic electromagnetic path.
    # Create array to get the correct index for adjacent points
    x_adj = np.zeros((electrons.N, 2), dtype=int)
    y_adj = np.zeros((electrons.N, 2), dtype=int)
    x_adj[:, 0] = 1
    y_adj[:, 1] = 1
    np.add.at(
        grid.J,
        (electrons.idx[:, 0], electrons.idx[:, 1]),
        electrons.v * electrons.q * (1 - electrons.cic_weights[:, :1]) * (1 - electrons.cic_weights[:, 1:]),
    )
    coord = (electrons.idx + x_adj) % grid.n_cells
    np.add.at(grid.J, (coord[:, 0], coord[:, 1]), electrons.v * electrons.q * electrons.cic_weights[:, :1] * (1 - electrons.cic_weights[:, 1:]))
    coord = (electrons.idx + y_adj) % grid.n_cells
    np.add.at(grid.J, (coord[:, 0], coord[:, 1]), electrons.v * electrons.q * electrons.cic_weights[:, 1:] * (1 - electrons.cic_weights[:, :1]))
    coord = (electrons.idx + x_adj + y_adj) % grid.n_cells
    np.add.at(grid.J, (coord[:, 0], coord[:, 1]), electrons.v * electrons.q * electrons.cic_weights[:, :1] * electrons.cic_weights[:, 1:])

    x_adj = np.zeros((ions.N, 2), dtype=int)
    y_adj = np.zeros((ions.N, 2), dtype=int)
    x_adj[:, 0] = 1
    y_adj[:, 1] = 1
    np.add.at(grid.J, (ions.idx[:, 0], ions.idx[:, 1]), ions.v * ions.q * (1 - ions.cic_weights[:, :1]) * (1 - ions.cic_weights[:, 1:]))
    coord = (ions.idx + x_adj) % grid.n_cells
    np.add.at(grid.J, (coord[:, 0], coord[:, 1]), ions.v * ions.q * ions.cic_weights[:, :1] * (1 - ions.cic_weights[:, 1:]))
    coord = (ions.idx + y_adj) % grid.n_cells
    np.add.at(grid.J, (coord[:, 0], coord[:, 1]), ions.v * ions.q * ions.cic_weights[:, 1:] * (1 - ions.cic_weights[:, :1]))
    coord = (ions.idx + x_adj + y_adj) % grid.n_cells
    np.add.at(grid.J, (coord[:, 0], coord[:, 1]), ions.v * ions.q * ions.cic_weights[:, :1] * ions.cic_weights[:, 1:])


def calc_E_2D(grid: Grid2D, dt, bc):
    raise RuntimeError(
        "Legacy split 2D field update is disabled; use calc_fields_2D"
    )
    """
    Maxwell's equations for 2D become:

    dE_x/dt = -J_x / eps_0 + c² dB_z/dy\n
    dE_y/dt = -J_y / eps_0 - c² dB_z/dx\n
    dB_z/dt = dE_x/dy - dE_y/dx\n
    """
    # Solve Euler's equation to find E
    # euler_solver_2D(grid, dt, bc)

    if bc is BoundaryCondition.Periodic:
        # calculate the fields at the full timestep
        # np.roll(Ez, -1) = [Ez(x1), Ez(x2), ..., Ez(xN), Ez(x0)]
        grid.E[:, :, 0] += dt * (-grid.J[:, :, 0] / eps_0 + c * c / grid.dx * (grid.B[:, :, 0] - np.roll(grid.B[:, :, 0], 1, axis=0)))
        grid.E[:, :, 1] += dt * (-grid.J[:, :, 1] / eps_0 - c * c / grid.dx * (grid.B[:, :, 0] - np.roll(grid.B[:, :, 0], 1, axis=1)))
    else:
        # calculate the fields at the full timestep
        grid.E[1:, :, 0] += dt * (-grid.J[1:, :, 0] / eps_0 + c * c / grid.dx * (grid.B[1:, :, 0] - np.roll(grid.B[:-1, :, 0], 1, axis=0)))
        grid.E[:, 1:, 1] += dt * (-grid.J[:, 1:, 1] / eps_0 - c * c / grid.dx * (grid.B[:, 1:, 0] - np.roll(grid.B[:, :-1, 0], 1, axis=1)))

        # calculate the boundary values using interpolation
        grid.E[0, :, 0] = 3 * grid.E[1, :, 0] - 3 * grid.E[2, :, 0] + grid.E[3, :, 0]
        grid.E[0, :, 1] = 3 * grid.E[:, 1, 1] - 3 * grid.E[:, 2, 1] + grid.E[:, 3, 1]


def calc_B_2D(grid: Grid2D, dt, bc):
    raise RuntimeError(
        "Legacy split 2D field update is disabled; use calc_fields_2D"
    )
    """
    dB_z/dt = dE_x/dy - dE_y/dx\n
    """
    if bc is BoundaryCondition.Periodic:
        # calculate the fields at the full timestep
        # np.roll(Ez, -1) = [Ez(x1), Ez(x2), ..., Ez(xN), Ez(x0)]
        grid.B[:, :, 0] += (
            dt
            * c
            * c
            * (
                -(np.roll(grid.E[:, :, 1], -1, axis=1) - grid.E[:, :, 1]) / grid.dx
                + (np.roll(grid.E[:, :, 0], -1, axis=0) - grid.E[:, :, 0]) / grid.dx
            )
        )
    else:
        # calculate the fields at the full timestep
        grid.B[:-1, :-1, 0] += (
            dt * c * c * (-(grid.E[1:, 1:, 1] - grid.E[:-1, :-1, 1]) / grid.dx + (grid.E[1:, 1:, 0] - grid.E[:-1, :-1, 0]) / grid.dx)
        )

        # calculate the boundary values using interpolation
        grid.B[-1, :-1, 0] = 3 * grid.B[-2, :-1, 0] - 3 * grid.B[-3, :-1, 0] + grid.B[-4, :-1, 0]
        grid.B[:-1, -1, 0] = 3 * grid.B[:-1, -2, 0] - 3 * grid.B[:-1, -3, 0] + grid.B[:-1, -4, 0]
        grid.B[-1, -1, 0] = 3 * grid.B[-1, -2, 0] - 3 * grid.B[-1, -3, 0] + grid.B[-1, -4, 0]  # last value is an interpolation of an interpolation


# Validated periodic 2D2V TMz path. The legacy calc_E_2D/calc_B_2D routines
# above are retained only for historical comparison and are not used by the
# repaired solver.
def calc_fields_2D(grid: Grid2D, dt):
    """Stable periodic TMz Maxwell update on a 2D Yee grid."""
    half_dt = 0.5 * dt
    bz = grid.B[:, :, 0]

    bz += half_dt * (
        (np.roll(grid.E[:, :, 0], -1, axis=1) - grid.E[:, :, 0]) / grid.dx
        - (np.roll(grid.E[:, :, 1], -1, axis=0) - grid.E[:, :, 1]) / grid.dx
    )
    grid.E[:, :, 0] += dt * (
        -grid.J[:, :, 0] / eps_0
        + c * c * (bz - np.roll(bz, 1, axis=1)) / grid.dx
    )
    grid.E[:, :, 1] += dt * (
        -grid.J[:, :, 1] / eps_0
        - c * c * (bz - np.roll(bz, 1, axis=0)) / grid.dx
    )
    bz += half_dt * (
        (np.roll(grid.E[:, :, 0], -1, axis=1) - grid.E[:, :, 0]) / grid.dx
        - (np.roll(grid.E[:, :, 1], -1, axis=0) - grid.E[:, :, 1]) / grid.dx
    )


def calc_gauss_residual_2D(grid: Grid2D):
    divergence = (
        (grid.E[:, :, 0] - np.roll(grid.E[:, :, 0], 1, axis=0)) / grid.dx
        + (grid.E[:, :, 1] - np.roll(grid.E[:, :, 1], 1, axis=1)) / grid.dx
    )
    return divergence - grid.rho / eps_0


def initialize_electric_field_2D(grid: Grid2D):
    """Solve periodic discrete Gauss law for the minimum-energy E field."""
    net_charge = grid.dx * grid.dx * np.sum(grid.rho)
    scale = grid.dx * grid.dx * np.sum(np.abs(grid.rho))
    if abs(net_charge) > 1e-12 * max(1.0, scale):
        raise ValueError("A periodic 2D domain must be charge neutral")

    rho_hat = np.fft.fft2(grid.rho)
    modes = 2 * np.pi * np.fft.fftfreq(grid.n_cells, d=grid.dx)
    symbol = 2 * np.sin(0.5 * modes * grid.dx) / grid.dx
    sx = symbol[:, np.newaxis]
    sy = symbol[np.newaxis, :]
    denom = sx * sx + sy * sy
    potential_hat = np.zeros_like(rho_hat, dtype=complex)
    mask = denom > 0
    potential_hat[mask] = rho_hat[mask] / (eps_0 * denom[mask])
    phase_x = np.exp(0.5j * modes * grid.dx)[:, np.newaxis]
    phase_y = np.exp(0.5j * modes * grid.dx)[np.newaxis, :]
    grid.E[:, :, 0] = np.fft.ifft2(-1j * sx * phase_x * potential_hat).real
    grid.E[:, :, 1] = np.fft.ifft2(-1j * sy * phase_y * potential_hat).real
    grid.gauss_residual[:] = calc_gauss_residual_2D(grid)


def _deposit_midpoint_current_component_2D(grid, target, particles, midpoint, component, offset):
    scaled = midpoint / grid.dx - np.asarray(offset)
    idx = np.floor(scaled).astype(np.int32)
    weight = scaled - idx
    ix = idx[:, 0] % grid.n_cells
    iy = idx[:, 1] % grid.n_cells
    wx = weight[:, 0]
    wy = weight[:, 1]
    value = particles.q * particles.weight / (grid.dx * grid.dx) * particles.v[:, component]
    np.add.at(target, (ix, iy), value * (1 - wx) * (1 - wy))
    np.add.at(target, ((ix + 1) % grid.n_cells, iy), value * wx * (1 - wy))
    np.add.at(target, (ix, (iy + 1) % grid.n_cells), value * (1 - wx) * wy)
    np.add.at(
        target,
        ((ix + 1) % grid.n_cells, (iy + 1) % grid.n_cells),
        value * wx * wy,
    )


def calc_charge_conserving_current_2D(
    grid: Grid2D,
    electrons: Particles,
    ions: Particles,
    electron_x_old,
    ion_x_old,
    rho_old,
    dt,
):
    """Deposit midpoint current and project it onto discrete continuity.

    The spectral correction changes only the curl-free part of J and enforces
    (rho_new-rho_old)/dt + div(J) = 0 to roundoff on the periodic Yee grid.
    """
    grid.J.fill(0)
    for particles, old_position in ((electrons, electron_x_old), (ions, ion_x_old)):
        displacement = (
            particles.x - old_position + 0.5 * grid.x_max
        ) % grid.x_max - 0.5 * grid.x_max
        midpoint = (old_position + 0.5 * displacement) % grid.x_max
        _deposit_midpoint_current_component_2D(
            grid, grid.J[:, :, 0], particles, midpoint, 0, (0.5, 0.0)
        )
        _deposit_midpoint_current_component_2D(
            grid, grid.J[:, :, 1], particles, midpoint, 1, (0.0, 0.5)
        )

    delta_rho = grid.rho - rho_old
    residual = delta_rho / dt + (
        (grid.J[:, :, 0] - np.roll(grid.J[:, :, 0], 1, axis=0)) / grid.dx
        + (grid.J[:, :, 1] - np.roll(grid.J[:, :, 1], 1, axis=1)) / grid.dx
    )
    residual_hat = np.fft.fft2(residual)
    modes = 2 * np.pi * np.fft.fftfreq(grid.n_cells, d=grid.dx)
    symbol = 2 * np.sin(0.5 * modes * grid.dx) / grid.dx
    sx = symbol[:, np.newaxis]
    sy = symbol[np.newaxis, :]
    denom = sx * sx + sy * sy
    potential_hat = np.zeros_like(residual_hat, dtype=complex)
    mask = denom > 0
    potential_hat[mask] = residual_hat[mask] / denom[mask]
    phase_x = np.exp(0.5j * modes * grid.dx)[:, np.newaxis]
    phase_y = np.exp(0.5j * modes * grid.dx)[np.newaxis, :]
    grid.J[:, :, 0] += np.fft.ifft2(1j * sx * phase_x * potential_hat).real
    grid.J[:, :, 1] += np.fft.ifft2(1j * sy * phase_y * potential_hat).real
    grid.continuity_residual[:] = delta_rho / dt + (
        (grid.J[:, :, 0] - np.roll(grid.J[:, :, 0], 1, axis=0)) / grid.dx
        + (grid.J[:, :, 1] - np.roll(grid.J[:, :, 1], 1, axis=1)) / grid.dx
    )
