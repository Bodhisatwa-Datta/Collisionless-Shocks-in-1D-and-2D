import numpy as np

from grids import Grid1D, Grid1D3V, Grid2D
import maxwell
from parameters import Parameters
from particles import Particles
from physical_constants import c


def advance_positions(particles: Particles, dt):
    particles.x += particles.v[:, : particles.dimX] * dt


def initialize_velocities_half_step_1D(
    grid: Grid1D,
    electrons: Particles,
    ions: Particles,
    params: Parameters,
    dt: float,
    tridiag,
):
    maxwell.poisson_solver(grid, electrons, ions, params, tridiag, first=True)
    lorenz_force_1D(grid, electrons, -dt / 2)
    lorenz_force_1D(grid, ions, -dt / 2)


def lorenz_force_1D(grid: Grid1D, particles: Particles, dt):
    electric = (
        grid.E[particles.idx] * (1 - particles.cic_weights)
        + grid.E[(particles.idx + 1) % grid.n_cells] * particles.cic_weights
    )
    particles.v += particles.qm * electric * dt


def initialize_velocities_half_step_1D3V(
    grid: Grid1D3V,
    electrons: Particles,
    ions: Particles,
    params: Parameters,
    dt: float,
):
    boris_pusher_1D3V(grid, electrons, dt / 2)
    boris_pusher_1D3V(grid, ions, dt / 2)


def boris_pusher_1D3V(grid: Grid1D3V, particles: Particles, dt):
    electric = (
        grid.E[particles.idx.flatten()] * (1 - particles.cic_weights)
        + grid.E[(particles.idx.flatten() + 1) % grid.n_cells]
        * particles.cic_weights
    )
    magnetic = (
        grid.B[particles.idx_staggered.flatten()]
        * (1 - particles.cic_weights_staggered)
        + grid.B[(particles.idx_staggered.flatten() + 1) % grid.n_cells]
        * particles.cic_weights_staggered
    )
    ct = particles.qm * dt / 2
    u_minus = particles.u + ct * electric
    gamma_minus = np.sqrt(
        1 + np.sum(u_minus**2, axis=1, keepdims=True) / (c * c)
    )
    rotation = ct * magnetic / gamma_minus
    rotation_sq = np.sum(rotation * rotation, axis=1, keepdims=True)
    scale = 2 * rotation / (1 + rotation_sq)
    u_prime = u_minus + np.cross(u_minus, rotation)
    u_plus = u_minus + np.cross(u_prime, scale)
    particles.u = u_plus + ct * electric
    gamma_new = np.sqrt(
        1 + np.sum(particles.u**2, axis=1, keepdims=True) / (c * c)
    )
    particles.v = particles.u / gamma_new


def initialize_velocities_half_step_2D(
    grid: Grid2D,
    electrons: Particles,
    ions: Particles,
    params: Parameters,
    dt: float,
):
    boris_pusher_2D2V(grid, electrons, dt / 2)
    boris_pusher_2D2V(grid, ions, dt / 2)


def _interpolate_2D(field, positions, dx, offset):
    n_cells = field.shape[0]
    scaled = positions / dx - np.asarray(offset)
    idx = np.floor(scaled).astype(np.int32)
    weight = scaled - idx
    ix = idx[:, 0] % n_cells
    iy = idx[:, 1] % n_cells
    wx = weight[:, 0]
    wy = weight[:, 1]
    return (
        field[ix, iy] * (1 - wx) * (1 - wy)
        + field[(ix + 1) % n_cells, iy] * wx * (1 - wy)
        + field[ix, (iy + 1) % n_cells] * (1 - wx) * wy
        + field[(ix + 1) % n_cells, (iy + 1) % n_cells] * wx * wy
    )


def boris_pusher_2D2V(grid: Grid2D, particles: Particles, dt):
    """Relativistic 2D2V Boris update for Ex, Ey, and Bz on a Yee grid."""
    electric = np.column_stack(
        (
            _interpolate_2D(grid.E[:, :, 0], particles.x, grid.dx, (0.5, 0.0)),
            _interpolate_2D(grid.E[:, :, 1], particles.x, grid.dx, (0.0, 0.5)),
        )
    )
    magnetic = _interpolate_2D(
        grid.B[:, :, 0], particles.x, grid.dx, (0.5, 0.5)
    )[:, np.newaxis]
    ct = particles.qm * dt / 2
    u_minus = particles.u + ct * electric
    gamma_minus = np.sqrt(
        1 + np.sum(u_minus**2, axis=1, keepdims=True) / (c * c)
    )
    rotation = ct * magnetic / gamma_minus
    scale = 2 * rotation / (1 + rotation * rotation)
    u_prime = u_minus + np.column_stack(
        (u_minus[:, 1] * rotation[:, 0], -u_minus[:, 0] * rotation[:, 0])
    )
    u_plus = u_minus + np.column_stack(
        (u_prime[:, 1] * scale[:, 0], -u_prime[:, 0] * scale[:, 0])
    )
    particles.u = u_plus + ct * electric
    gamma_new = np.sqrt(
        1 + np.sum(particles.u**2, axis=1, keepdims=True) / (c * c)
    )
    particles.v = particles.u / gamma_new


# Backward-compatible name for callers outside solver_2D.py.
boris_pusher_2D = boris_pusher_2D2V
