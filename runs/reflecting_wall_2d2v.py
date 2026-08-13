"""Electrostatic 2D2V reflecting-wall collisionless-shock experiment.

The x boundaries are specularly reflecting and the y boundary is periodic.
Poisson's equation uses an even extension in x (homogeneous Neumann wall
condition) and a periodic FFT in the doubled domain.
"""

from dataclasses import dataclass, field
import sys

sys.path.append("./")
sys.path.append("../")

import numpy as np

from particles import Particles
from physical_constants import c, eps_0, m_e, q_e, q_i


LENGTH_X = 40.0
LENGTH_Y = 6.0
N_X = 128
N_Y = 24
PARTICLES_X = 128
PARTICLES_Y = 48
ION_MASS = 100.0
INFLOW_SPEED = -0.03
ELECTRON_THERMAL_SPEED = 0.10
ION_THERMAL_SPEED = 0.002
DT = 0.01
T_MAX = 120.0
SAVE_INTERVAL = 100
SEED = 41


@dataclass
class Results2D2V:
    x_grid: np.ndarray
    y_grid: np.ndarray
    t: list = field(default_factory=list)
    n_e: list = field(default_factory=list)
    n_i: list = field(default_factory=list)
    electric: list = field(default_factory=list)
    ion_x: list = field(default_factory=list)
    ion_v: list = field(default_factory=list)
    total_energy: list = field(default_factory=list)
    gauss_linf: list = field(default_factory=list)
    final_electron_x: np.ndarray | None = None
    final_electron_v: np.ndarray | None = None


def initialize_particles(seed=SEED):
    rng = np.random.default_rng(seed)
    x = (np.arange(PARTICLES_X) + 0.5) * LENGTH_X / PARTICLES_X
    y = (np.arange(PARTICLES_Y) + 0.5) * LENGTH_Y / PARTICLES_Y
    xx, yy = np.meshgrid(x, y, indexing="ij")
    positions = np.column_stack((xx.ravel(), yy.ravel()))
    count = len(positions)
    electrons = Particles(count, 2, 2, m_e, q_e)
    ions = Particles(count, 2, 2, ION_MASS, q_i)
    electrons.x[:] = positions
    ions.x[:] = positions
    electrons.v[:, 0] = rng.normal(INFLOW_SPEED, ELECTRON_THERMAL_SPEED, count)
    electrons.v[:, 1] = rng.normal(0.0, ELECTRON_THERMAL_SPEED, count)
    ions.v[:, 0] = rng.normal(INFLOW_SPEED, ION_THERMAL_SPEED, count)
    ions.v[:, 1] = rng.normal(0.0, ION_THERMAL_SPEED, count)
    electrons.v[:, 1] -= np.mean(electrons.v[:, 1])
    ions.v[:, 1] -= np.mean(ions.v[:, 1])
    electrons.v_to_u()
    ions.v_to_u()
    weight = LENGTH_X * LENGTH_Y / count
    electrons.weight = weight
    ions.weight = weight
    return electrons, ions


def _cell_coordinates(positions, dx, dy):
    sx = positions[:, 0] / dx - 0.5
    raw_x = np.floor(sx).astype(np.int32)
    ix = np.clip(raw_x, 0, N_X - 2)
    fx = np.clip(sx - raw_x, 0.0, 1.0)
    fx[raw_x < 0] = 0.0
    fx[raw_x >= N_X - 1] = 1.0

    sy = positions[:, 1] / dy - 0.5
    raw_y = np.floor(sy).astype(np.int32)
    iy = raw_y % N_Y
    fy = sy - raw_y
    return ix, iy, fx, fy


def deposit_density(particles, dx, dy):
    density = np.zeros((N_X, N_Y))
    ix, iy, fx, fy = _cell_coordinates(particles.x, dx, dy)
    scale = particles.weight / (dx * dy)
    np.add.at(density, (ix, iy), scale * (1 - fx) * (1 - fy))
    np.add.at(density, (ix + 1, iy), scale * fx * (1 - fy))
    np.add.at(density, (ix, (iy + 1) % N_Y), scale * (1 - fx) * fy)
    np.add.at(density, (ix + 1, (iy + 1) % N_Y), scale * fx * fy)
    return density


def solve_field(electrons, ions, dx, dy):
    n_e = deposit_density(electrons, dx, dy)
    n_i = deposit_density(ions, dx, dy)
    rho = q_e * n_e + q_i * n_i

    # An even extension around both x walls imposes d(phi)/dx=0 there.
    rho_extended = np.concatenate((rho, rho[::-1, :]), axis=0)
    rho_hat = np.fft.fftn(rho_extended)
    kx_index = np.arange(2 * N_X)
    ky_index = np.arange(N_Y)
    lap_x = -4.0 * np.sin(np.pi * kx_index / (2 * N_X)) ** 2 / dx**2
    lap_y = -4.0 * np.sin(np.pi * ky_index / N_Y) ** 2 / dy**2
    eigenvalue = lap_x[:, None] + lap_y[None, :]
    phi_hat = np.zeros_like(rho_hat)
    nonzero = eigenvalue != 0.0
    phi_hat[nonzero] = -rho_hat[nonzero] / (eps_0 * eigenvalue[nonzero])
    phi = np.fft.ifftn(phi_hat).real[:N_X, :]

    ex_faces = np.zeros((N_X + 1, N_Y))
    ex_faces[1:N_X] = -(phi[1:] - phi[:-1]) / dx
    ey_faces = -(phi - np.roll(phi, 1, axis=1)) / dy
    ex = 0.5 * (ex_faces[:-1] + ex_faces[1:])
    ey = 0.5 * (ey_faces + np.roll(ey_faces, -1, axis=1))
    electric = np.stack((ex, ey), axis=2)
    divergence = (
        (ex_faces[1:] - ex_faces[:-1]) / dx
        + (np.roll(ey_faces, -1, axis=1) - ey_faces) / dy
    )
    gauss = divergence - rho / eps_0
    return n_e, n_i, rho, electric, float(np.max(np.abs(gauss)))


def interpolate_field(particles, electric, dx, dy):
    ix, iy, fx, fy = _cell_coordinates(particles.x, dx, dy)
    return (
        electric[ix, iy] * ((1 - fx) * (1 - fy))[:, None]
        + electric[ix + 1, iy] * (fx * (1 - fy))[:, None]
        + electric[ix, (iy + 1) % N_Y] * ((1 - fx) * fy)[:, None]
        + electric[ix + 1, (iy + 1) % N_Y] * (fx * fy)[:, None]
    )


def electric_kick(particles, electric, dx, dy, dt):
    particles.u += particles.qm * interpolate_field(particles, electric, dx, dy) * dt
    gamma = np.sqrt(1.0 + np.sum(particles.u**2, axis=1) / c**2)
    particles.v[:] = particles.u / gamma[:, None]


def drift_and_bound(particles, dt):
    particles.x += particles.v * dt
    left = particles.x[:, 0] < 0.0
    particles.x[left, 0] *= -1.0
    particles.u[left, 0] *= -1.0
    right = particles.x[:, 0] > LENGTH_X
    particles.x[right, 0] = 2.0 * LENGTH_X - particles.x[right, 0]
    particles.u[right, 0] *= -1.0
    particles.x[:, 1] %= LENGTH_Y
    gamma = np.sqrt(1.0 + np.sum(particles.u**2, axis=1) / c**2)
    particles.v[:] = particles.u / gamma[:, None]


def total_energy(electrons, ions, electric, dx, dy):
    kinetic = electrons.relativistic_kinetic_energy() + ions.relativistic_kinetic_energy()
    field = 0.5 * eps_0 * dx * dy * np.sum(electric**2)
    return float(kinetic + field)


def run():
    electrons, ions = initialize_particles()
    dx = LENGTH_X / N_X
    dy = LENGTH_Y / N_Y
    x_grid = (np.arange(N_X) + 0.5) * dx
    y_grid = (np.arange(N_Y) + 0.5) * dy
    results = Results2D2V(x_grid=x_grid, y_grid=y_grid)
    n_e, n_i, _, electric, gauss = solve_field(electrons, ions, dx, dy)

    def save(time):
        results.t.append(time)
        results.n_e.append(n_e.copy())
        results.n_i.append(n_i.copy())
        results.electric.append(electric.copy())
        results.ion_x.append(ions.x.copy())
        results.ion_v.append(ions.v.copy())
        results.total_energy.append(total_energy(electrons, ions, electric, dx, dy))
        results.gauss_linf.append(gauss)

    save(0.0)
    steps = int(np.ceil(T_MAX / DT))
    for step in range(1, steps + 1):
        electric_kick(electrons, electric, dx, dy, 0.5 * DT)
        electric_kick(ions, electric, dx, dy, 0.5 * DT)
        drift_and_bound(electrons, DT)
        drift_and_bound(ions, DT)
        n_e, n_i, _, electric, gauss = solve_field(electrons, ions, dx, dy)
        electric_kick(electrons, electric, dx, dy, 0.5 * DT)
        electric_kick(ions, electric, dx, dy, 0.5 * DT)
        if step % SAVE_INTERVAL == 0 or step == steps:
            save(step * DT)
    results.final_electron_x = electrons.x.copy()
    results.final_electron_v = electrons.v.copy()
    return results


if __name__ == "__main__":
    output = run()
    energy = np.asarray(output.total_energy)
    print("snapshots:", len(output.t))
    print("maximum relative energy drift:", np.max(np.abs((energy-energy[0])/energy[0])))
    print("maximum Gauss residual:", max(output.gauss_linf))
