"""Electrostatic 1D3V reflecting-wall collisionless-shock experiment.

Particles have three momentum/velocity components while the self-consistent
field is longitudinal Ex(x).  A cold ion/electron plasma flows toward x=0 and
is specularly reflected there.  The right wall is placed far enough away that
it does not affect the shock during the declared run time.
"""

from dataclasses import dataclass, field
import sys

sys.path.append("./")
sys.path.append("../")

import numpy as np

from particles import Particles
from physical_constants import c, eps_0, m_e, q_e, q_i


LENGTH = 40.0
N_CELLS = 400
PARTICLES_PER_SPECIES = 12_000
ION_MASS = 100.0
INFLOW_SPEED = -0.03
ELECTRON_THERMAL_SPEED = 0.10
ION_THERMAL_SPEED = 0.002
DT = 0.01
T_MAX = 120.0
SAVE_INTERVAL = 100
SEED = 31


@dataclass
class WallResults:
    x_grid: np.ndarray
    t: list = field(default_factory=list)
    n_e: list = field(default_factory=list)
    n_i: list = field(default_factory=list)
    ex: list = field(default_factory=list)
    ion_x: list = field(default_factory=list)
    ion_v: list = field(default_factory=list)
    total_energy: list = field(default_factory=list)
    gauss_linf: list = field(default_factory=list)


def initialize_particles(seed=SEED):
    rng = np.random.default_rng(seed)
    count = PARTICLES_PER_SPECIES
    positions = ((np.arange(count) + 0.5) * LENGTH / count).reshape(-1, 1)
    electrons = Particles(count, 1, 3, m_e, q_e)
    ions = Particles(count, 1, 3, ION_MASS, q_i)
    electrons.x[:] = positions
    ions.x[:] = positions
    electrons.v.fill(0.0)
    ions.v.fill(0.0)
    electrons.v[:, 0] = rng.normal(INFLOW_SPEED, ELECTRON_THERMAL_SPEED, count)
    ions.v[:, 0] = rng.normal(INFLOW_SPEED, ION_THERMAL_SPEED, count)
    electrons.v_to_u()
    ions.v_to_u()
    weight = LENGTH / count
    electrons.weight = weight
    ions.weight = weight
    return electrons, ions


def deposit_number_density(particles, dx, cells):
    """CIC deposition to cell centers without periodic wraparound."""
    density = np.zeros(cells)
    scaled = particles.x[:, 0] / dx - 0.5
    raw_index = np.floor(scaled).astype(np.int32)
    index = np.clip(raw_index, 0, cells - 2)
    fraction = np.clip(scaled - raw_index, 0.0, 1.0)
    scale = particles.weight / dx
    np.add.at(density, index, scale * (1.0 - fraction))
    np.add.at(density, index + 1, scale * fraction)
    # Outside the center-to-center interval, assign the complete shape to the
    # nearest boundary cell so every macro-particle retains its full weight.
    left = raw_index < 0
    if np.any(left):
        density[0] += scale * np.sum(fraction[left])
        density[1] -= scale * np.sum(fraction[left])
    right = raw_index >= cells - 1
    if np.any(right):
        density[-1] += scale * np.sum(1.0 - fraction[right])
        density[-2] -= scale * np.sum(1.0 - fraction[right])
    return density


def solve_field(electrons, ions, dx, cells):
    n_e = deposit_number_density(electrons, dx, cells)
    n_i = deposit_number_density(ions, dx, cells)
    rho = q_e * n_e + q_i * n_i
    ex_faces = np.zeros(cells + 1)
    ex_faces[1:] = dx * np.cumsum(rho) / eps_0
    ex = 0.5 * (ex_faces[:-1] + ex_faces[1:])
    gauss = (ex_faces[1:] - ex_faces[:-1]) / dx - rho / eps_0
    return n_e, n_i, rho, ex, float(np.max(np.abs(gauss)))


def interpolate_field(particles, ex, dx):
    scaled = particles.x[:, 0] / dx - 0.5
    index = np.floor(scaled).astype(np.int32)
    index = np.clip(index, 0, len(ex) - 2)
    fraction = np.clip(scaled - index, 0.0, 1.0)
    return ex[index] * (1.0 - fraction) + ex[index + 1] * fraction


def electric_kick(particles, ex, dx, dt):
    particles.u[:, 0] += particles.qm * interpolate_field(particles, ex, dx) * dt
    gamma = np.sqrt(1.0 + np.sum(particles.u**2, axis=1) / c**2)
    particles.v[:] = particles.u / gamma[:, None]


def drift_and_reflect(particles, dt):
    particles.x[:, 0] += particles.v[:, 0] * dt
    left = particles.x[:, 0] < 0.0
    particles.x[left, 0] *= -1.0
    particles.u[left, 0] *= -1.0
    right = particles.x[:, 0] > LENGTH
    particles.x[right, 0] = 2.0 * LENGTH - particles.x[right, 0]
    particles.u[right, 0] *= -1.0
    gamma = np.sqrt(1.0 + np.sum(particles.u**2, axis=1) / c**2)
    particles.v[:] = particles.u / gamma[:, None]


def total_energy(electrons, ions, ex, dx):
    particle = electrons.relativistic_kinetic_energy() + ions.relativistic_kinetic_energy()
    field_energy = 0.5 * eps_0 * dx * np.sum(ex**2)
    return float(particle + field_energy)


def run():
    electrons, ions = initialize_particles()
    dx = LENGTH / N_CELLS
    x_grid = (np.arange(N_CELLS) + 0.5) * dx
    results = WallResults(x_grid=x_grid)
    n_e, n_i, _, ex, gauss = solve_field(electrons, ions, dx, N_CELLS)

    def save(time):
        results.t.append(time)
        results.n_e.append(n_e.copy())
        results.n_i.append(n_i.copy())
        results.ex.append(ex.copy())
        results.ion_x.append(ions.x[:, 0].copy())
        results.ion_v.append(ions.v.copy())
        results.total_energy.append(total_energy(electrons, ions, ex, dx))
        results.gauss_linf.append(gauss)

    save(0.0)
    steps = int(np.ceil(T_MAX / DT))
    for step in range(1, steps + 1):
        electric_kick(electrons, ex, dx, 0.5 * DT)
        electric_kick(ions, ex, dx, 0.5 * DT)
        drift_and_reflect(electrons, DT)
        drift_and_reflect(ions, DT)
        n_e, n_i, _, ex, gauss = solve_field(electrons, ions, dx, N_CELLS)
        electric_kick(electrons, ex, dx, 0.5 * DT)
        electric_kick(ions, ex, dx, 0.5 * DT)
        if step % SAVE_INTERVAL == 0 or step == steps:
            save(step * DT)
    return results


if __name__ == "__main__":
    output = run()
    drift = np.max(
        np.abs((np.asarray(output.total_energy) - output.total_energy[0]) / output.total_energy[0])
    )
    print("snapshots:", len(output.t))
    print("maximum relative energy drift:", drift)
    print("maximum Gauss residual:", max(output.gauss_linf))
