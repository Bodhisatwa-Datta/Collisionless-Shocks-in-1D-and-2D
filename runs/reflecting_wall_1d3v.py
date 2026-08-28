"""Electrostatic 1D3V reflecting-wall collisionless-shock experiment.

Particles have three momentum/velocity components while the self-consistent
field is longitudinal Ex(x).  A cold ion/electron plasma flows toward x=0 and
is specularly reflected there.  The right wall is placed far enough away that
it does not affect the shock during the declared run time.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from pic.constants import c, eps_0, m_e, q_e, q_i
from pic.checkpoint import load_checkpoint, save_checkpoint
from pic.particles import Particles


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


@dataclass(frozen=True)
class WallConfig:
    """Complete, serializable configuration for a reflecting-wall run."""

    length: float = LENGTH
    n_cells: int = N_CELLS
    particles_per_species: int = PARTICLES_PER_SPECIES
    ion_mass: float = ION_MASS
    inflow_speed: float = INFLOW_SPEED
    electron_thermal_speed: float = ELECTRON_THERMAL_SPEED
    ion_thermal_speed: float = ION_THERMAL_SPEED
    dt: float = DT
    t_max: float = T_MAX
    save_interval: int = SAVE_INTERVAL
    seed: int = SEED

    def __post_init__(self):
        positive = {
            "length": self.length,
            "n_cells": self.n_cells,
            "particles_per_species": self.particles_per_species,
            "ion_mass": self.ion_mass,
            "dt": self.dt,
            "t_max": self.t_max,
            "save_interval": self.save_interval,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.n_cells < 2:
            raise ValueError("n_cells must be at least two")

    @property
    def dx(self):
        return self.length / self.n_cells

    @property
    def steps(self):
        return int(np.ceil(self.t_max / self.dt))


@dataclass
class WallResults:
    x_grid: np.ndarray
    t: list = field(default_factory=list)
    n_e: list = field(default_factory=list)
    n_i: list = field(default_factory=list)
    ex: list = field(default_factory=list)
    ion_x: list = field(default_factory=list)
    ion_v: list = field(default_factory=list)
    electron_kinetic_energy: list = field(default_factory=list)
    ion_kinetic_energy: list = field(default_factory=list)
    field_energy: list = field(default_factory=list)
    total_energy: list = field(default_factory=list)
    gauss_linf: list = field(default_factory=list)
    final_electron_x: np.ndarray | None = None
    final_electron_v: np.ndarray | None = None
    configuration: dict = field(default_factory=dict)


def initialize_particles(config=WallConfig()):
    rng = np.random.default_rng(config.seed)
    count = config.particles_per_species
    positions = ((np.arange(count) + 0.5) * config.length / count).reshape(-1, 1)
    electrons = Particles(count, 1, 3, m_e, q_e)
    ions = Particles(count, 1, 3, config.ion_mass, q_i)
    electrons.x[:] = positions
    ions.x[:] = positions
    electrons.v.fill(0.0)
    ions.v.fill(0.0)
    electrons.v[:, 0] = rng.normal(
        config.inflow_speed, config.electron_thermal_speed, count
    )
    ions.v[:, 0] = rng.normal(config.inflow_speed, config.ion_thermal_speed, count)
    electrons.v_to_u()
    ions.v_to_u()
    weight = config.length / count
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


def drift_and_reflect(particles, dt, length=LENGTH):
    particles.x[:, 0] += particles.v[:, 0] * dt
    left = particles.x[:, 0] < 0.0
    particles.x[left, 0] *= -1.0
    particles.u[left, 0] *= -1.0
    right = particles.x[:, 0] > length
    particles.x[right, 0] = 2.0 * length - particles.x[right, 0]
    particles.u[right, 0] *= -1.0
    gamma = np.sqrt(1.0 + np.sum(particles.u**2, axis=1) / c**2)
    particles.v[:] = particles.u / gamma[:, None]


def energy_components(electrons, ions, ex, dx):
    """Return electron, ion, field, and total energy in normalized units."""

    electron = float(electrons.relativistic_kinetic_energy())
    ion = float(ions.relativistic_kinetic_energy())
    field = float(0.5 * eps_0 * dx * np.sum(ex**2))
    return electron, ion, field, electron + ion + field


def total_energy(electrons, ions, ex, dx):
    return energy_components(electrons, ions, ex, dx)[-1]


def _particle_from_state(x, u, mass, charge, weight):
    particles = Particles(len(x), x.shape[1], u.shape[1], mass, charge)
    particles.x[:] = x
    particles.u[:] = u
    gamma = np.sqrt(1.0 + np.sum(u**2, axis=1) / c**2)
    particles.v[:] = u / gamma[:, None]
    particles.weight = float(weight)
    return particles


def _write_run_checkpoint(path, config, step, electrons, ions, results):
    arrays = {
        "electron_x": electrons.x,
        "electron_u": electrons.u,
        "ion_x_state": ions.x,
        "ion_u": ions.u,
        "times": np.asarray(results.t),
        "n_e": np.asarray(results.n_e),
        "n_i": np.asarray(results.n_i),
        "ex": np.asarray(results.ex),
        "ion_x_history": np.asarray(results.ion_x),
        "ion_v_history": np.asarray(results.ion_v),
        "electron_kinetic_energy": np.asarray(results.electron_kinetic_energy),
        "ion_kinetic_energy": np.asarray(results.ion_kinetic_energy),
        "field_energy": np.asarray(results.field_energy),
        "total_energy": np.asarray(results.total_energy),
        "gauss_linf": np.asarray(results.gauss_linf),
    }
    save_checkpoint(
        path,
        metadata={
            "simulation": "reflecting_wall_1d3v",
            "step": int(step),
            "configuration": asdict(config),
            "electron_weight": float(electrons.weight),
            "ion_weight": float(ions.weight),
        },
        arrays=arrays,
    )


def _read_run_checkpoint(path):
    metadata, arrays = load_checkpoint(path)
    if metadata.get("simulation") != "reflecting_wall_1d3v":
        raise ValueError("checkpoint belongs to a different simulation")
    config = WallConfig(**metadata["configuration"])
    electrons = _particle_from_state(
        arrays["electron_x"], arrays["electron_u"], m_e, q_e, metadata["electron_weight"]
    )
    ions = _particle_from_state(
        arrays["ion_x_state"], arrays["ion_u"], config.ion_mass, q_i, metadata["ion_weight"]
    )
    results = WallResults(
        x_grid=(np.arange(config.n_cells) + 0.5) * config.dx,
        t=list(arrays["times"]),
        n_e=list(arrays["n_e"]),
        n_i=list(arrays["n_i"]),
        ex=list(arrays["ex"]),
        ion_x=list(arrays["ion_x_history"]),
        ion_v=list(arrays["ion_v_history"]),
        electron_kinetic_energy=list(
            arrays.get("electron_kinetic_energy", np.full(len(arrays["times"]), np.nan))
        ),
        ion_kinetic_energy=list(
            arrays.get("ion_kinetic_energy", np.full(len(arrays["times"]), np.nan))
        ),
        field_energy=list(
            arrays.get("field_energy", np.full(len(arrays["times"]), np.nan))
        ),
        total_energy=list(arrays["total_energy"]),
        gauss_linf=list(arrays["gauss_linf"]),
        configuration=asdict(config),
    )
    return config, int(metadata["step"]), electrons, ions, results


def run(config=None, *, stop_time=None, checkpoint_path=None, resume_from=None):
    """Run, pause, or resume the reflecting-wall experiment.

    ``stop_time`` can end a run before ``config.t_max``. If ``checkpoint_path``
    is provided, the complete state and saved history are written at that
    point. Passing that file as ``resume_from`` continues to the configured
    final time without reinitializing the particles.
    """

    if resume_from is not None:
        stored_config, start_step, electrons, ions, results = _read_run_checkpoint(
            resume_from
        )
        if config is not None and config != stored_config:
            raise ValueError("the supplied configuration does not match the checkpoint")
        config = stored_config
    else:
        config = WallConfig() if config is None else config
        electrons, ions = initialize_particles(config)
        start_step = 0
        x_grid = (np.arange(config.n_cells) + 0.5) * config.dx
        results = WallResults(x_grid=x_grid, configuration=asdict(config))

    dx = config.dx
    n_e, n_i, _, ex, gauss = solve_field(
        electrons, ions, dx, config.n_cells
    )

    def save(time):
        results.t.append(time)
        results.n_e.append(n_e.copy())
        results.n_i.append(n_i.copy())
        results.ex.append(ex.copy())
        results.ion_x.append(ions.x[:, 0].copy())
        results.ion_v.append(ions.v.copy())
        electron_energy, ion_energy, electric_energy, total = energy_components(
            electrons, ions, ex, dx
        )
        results.electron_kinetic_energy.append(electron_energy)
        results.ion_kinetic_energy.append(ion_energy)
        results.field_energy.append(electric_energy)
        results.total_energy.append(total)
        results.gauss_linf.append(gauss)

    if start_step == 0 and not results.t:
        save(0.0)
    requested_time = config.t_max if stop_time is None else min(stop_time, config.t_max)
    target_step = min(config.steps, int(np.ceil(requested_time / config.dt)))
    if target_step < start_step:
        raise ValueError("stop_time precedes the checkpoint time")
    for step in range(start_step + 1, target_step + 1):
        electric_kick(electrons, ex, dx, 0.5 * config.dt)
        electric_kick(ions, ex, dx, 0.5 * config.dt)
        drift_and_reflect(electrons, config.dt, config.length)
        drift_and_reflect(ions, config.dt, config.length)
        n_e, n_i, _, ex, gauss = solve_field(
            electrons, ions, dx, config.n_cells
        )
        electric_kick(electrons, ex, dx, 0.5 * config.dt)
        electric_kick(ions, ex, dx, 0.5 * config.dt)
        if step % config.save_interval == 0 or step == target_step:
            save(step * config.dt)
    results.final_electron_x = electrons.x[:, 0].copy()
    results.final_electron_v = electrons.v.copy()
    if checkpoint_path is not None:
        _write_run_checkpoint(
            Path(checkpoint_path), config, target_step, electrons, ions, results
        )
    return results


if __name__ == "__main__":
    output = run()
    drift = np.max(
        np.abs((np.asarray(output.total_energy) - output.total_energy[0]) / output.total_energy[0])
    )
    print("snapshots:", len(output.t))
    print("maximum relative energy drift:", drift)
    print("maximum Gauss residual:", max(output.gauss_linf))
