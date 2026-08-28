"""Electrostatic 2D2V reflecting-wall collisionless-shock experiment.

The x boundaries are specularly reflecting and the y boundary is periodic.
Poisson's equation uses an even extension in x (homogeneous Neumann wall
condition) and a periodic FFT in the doubled domain.
"""

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from pic.checkpoint import load_checkpoint, save_checkpoint
from pic.constants import c, eps_0, m_e, q_e, q_i
from pic.particles import Particles


LENGTH_X = 20.0
LENGTH_Y = 4.0
N_X = 200
N_Y = 20
PARTICLES_X = 200
PARTICLES_Y = 40
ION_MASS = 100.0
INFLOW_SPEED = -0.03
ELECTRON_THERMAL_SPEED = 0.10
ION_THERMAL_SPEED = 0.002
DT = 0.01
T_MAX = 100.0
SAVE_INTERVAL = 100
SEED = 41


@dataclass(frozen=True)
class WallConfig2D:
    """Complete, serializable configuration for a 2D reflecting-wall run."""

    length_x: float = LENGTH_X
    length_y: float = LENGTH_Y
    n_x: int = N_X
    n_y: int = N_Y
    particles_x: int = PARTICLES_X
    particles_y: int = PARTICLES_Y
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
            "length_x": self.length_x,
            "length_y": self.length_y,
            "n_x": self.n_x,
            "n_y": self.n_y,
            "particles_x": self.particles_x,
            "particles_y": self.particles_y,
            "ion_mass": self.ion_mass,
            "dt": self.dt,
            "t_max": self.t_max,
            "save_interval": self.save_interval,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.n_x < 2 or self.n_y < 2:
            raise ValueError("n_x and n_y must be at least two")

    @property
    def dx(self):
        return self.length_x / self.n_x

    @property
    def dy(self):
        return self.length_y / self.n_y

    @property
    def steps(self):
        return int(np.ceil(self.t_max / self.dt))


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
    configuration: dict = field(default_factory=dict)


def initialize_particles(config=WallConfig2D()):
    rng = np.random.default_rng(config.seed)
    x = (np.arange(config.particles_x) + 0.5) * config.length_x / config.particles_x
    y = (np.arange(config.particles_y) + 0.5) * config.length_y / config.particles_y
    xx, yy = np.meshgrid(x, y, indexing="ij")
    positions = np.column_stack((xx.ravel(), yy.ravel()))
    count = len(positions)
    electrons = Particles(count, 2, 2, m_e, q_e)
    ions = Particles(count, 2, 2, config.ion_mass, q_i)
    electrons.x[:] = positions
    ions.x[:] = positions
    electrons.v[:, 0] = rng.normal(
        config.inflow_speed, config.electron_thermal_speed, count
    )
    electrons.v[:, 1] = rng.normal(0.0, config.electron_thermal_speed, count)
    ions.v[:, 0] = rng.normal(
        config.inflow_speed, config.ion_thermal_speed, count
    )
    ions.v[:, 1] = rng.normal(0.0, config.ion_thermal_speed, count)
    electrons.v[:, 1] -= np.mean(electrons.v[:, 1])
    ions.v[:, 1] -= np.mean(ions.v[:, 1])
    electrons.v_to_u()
    ions.v_to_u()
    weight = config.length_x * config.length_y / count
    electrons.weight = weight
    ions.weight = weight
    return electrons, ions


def _cell_coordinates(positions, dx, dy, n_x=N_X, n_y=N_Y):
    sx = positions[:, 0] / dx - 0.5
    raw_x = np.floor(sx).astype(np.int32)
    ix = np.clip(raw_x, 0, n_x - 2)
    fx = np.clip(sx - raw_x, 0.0, 1.0)
    fx[raw_x < 0] = 0.0
    fx[raw_x >= n_x - 1] = 1.0

    sy = positions[:, 1] / dy - 0.5
    raw_y = np.floor(sy).astype(np.int32)
    iy = raw_y % n_y
    fy = sy - raw_y
    return ix, iy, fx, fy


def deposit_density(particles, dx, dy, n_x=N_X, n_y=N_Y):
    density = np.zeros((n_x, n_y))
    ix, iy, fx, fy = _cell_coordinates(particles.x, dx, dy, n_x, n_y)
    scale = particles.weight / (dx * dy)
    np.add.at(density, (ix, iy), scale * (1 - fx) * (1 - fy))
    np.add.at(density, (ix + 1, iy), scale * fx * (1 - fy))
    np.add.at(density, (ix, (iy + 1) % n_y), scale * (1 - fx) * fy)
    np.add.at(density, (ix + 1, (iy + 1) % n_y), scale * fx * fy)
    return density


def solve_field(electrons, ions, dx, dy, n_x=N_X, n_y=N_Y):
    n_e = deposit_density(electrons, dx, dy, n_x, n_y)
    n_i = deposit_density(ions, dx, dy, n_x, n_y)
    rho = q_e * n_e + q_i * n_i

    # An even extension around both x walls imposes d(phi)/dx=0 there.
    rho_extended = np.concatenate((rho, rho[::-1, :]), axis=0)
    rho_hat = np.fft.fftn(rho_extended)
    kx_index = np.arange(2 * n_x)
    ky_index = np.arange(n_y)
    lap_x = -4.0 * np.sin(np.pi * kx_index / (2 * n_x)) ** 2 / dx**2
    lap_y = -4.0 * np.sin(np.pi * ky_index / n_y) ** 2 / dy**2
    eigenvalue = lap_x[:, None] + lap_y[None, :]
    phi_hat = np.zeros_like(rho_hat)
    nonzero = eigenvalue != 0.0
    phi_hat[nonzero] = -rho_hat[nonzero] / (eps_0 * eigenvalue[nonzero])
    phi = np.fft.ifftn(phi_hat).real[:n_x, :]

    ex_faces = np.zeros((n_x + 1, n_y))
    ex_faces[1:n_x] = -(phi[1:] - phi[:-1]) / dx
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


def interpolate_field(particles, electric, dx, dy, n_x=N_X, n_y=N_Y):
    ix, iy, fx, fy = _cell_coordinates(particles.x, dx, dy, n_x, n_y)
    return (
        electric[ix, iy] * ((1 - fx) * (1 - fy))[:, None]
        + electric[ix + 1, iy] * (fx * (1 - fy))[:, None]
        + electric[ix, (iy + 1) % n_y] * ((1 - fx) * fy)[:, None]
        + electric[ix + 1, (iy + 1) % n_y] * (fx * fy)[:, None]
    )


def electric_kick(particles, electric, dx, dy, dt, n_x=N_X, n_y=N_Y):
    particles.u += (
        particles.qm * interpolate_field(particles, electric, dx, dy, n_x, n_y) * dt
    )
    gamma = np.sqrt(1.0 + np.sum(particles.u**2, axis=1) / c**2)
    particles.v[:] = particles.u / gamma[:, None]


def drift_and_bound(particles, dt, length_x=LENGTH_X, length_y=LENGTH_Y):
    particles.x += particles.v * dt
    left = particles.x[:, 0] < 0.0
    particles.x[left, 0] *= -1.0
    particles.u[left, 0] *= -1.0
    right = particles.x[:, 0] > length_x
    particles.x[right, 0] = 2.0 * length_x - particles.x[right, 0]
    particles.u[right, 0] *= -1.0
    particles.x[:, 1] %= length_y
    gamma = np.sqrt(1.0 + np.sum(particles.u**2, axis=1) / c**2)
    particles.v[:] = particles.u / gamma[:, None]


def total_energy(electrons, ions, electric, dx, dy):
    kinetic = electrons.relativistic_kinetic_energy() + ions.relativistic_kinetic_energy()
    field = 0.5 * eps_0 * dx * dy * np.sum(electric**2)
    return float(kinetic + field)


def _particle_from_state(x, u, mass, charge, weight):
    particles = Particles(len(x), x.shape[1], u.shape[1], mass, charge)
    particles.x[:] = x
    particles.u[:] = u
    gamma = np.sqrt(1.0 + np.sum(u**2, axis=1) / c**2)
    particles.v[:] = u / gamma[:, None]
    particles.weight = float(weight)
    return particles


def _write_run_checkpoint(path, config, step, electrons, ions, results):
    save_checkpoint(
        path,
        metadata={
            "simulation": "reflecting_wall_2d2v",
            "step": int(step),
            "configuration": asdict(config),
            "electron_weight": float(electrons.weight),
            "ion_weight": float(ions.weight),
        },
        arrays={
            "electron_x": electrons.x,
            "electron_u": electrons.u,
            "ion_x_state": ions.x,
            "ion_u": ions.u,
            "times": np.asarray(results.t),
            "n_e": np.asarray(results.n_e),
            "n_i": np.asarray(results.n_i),
            "electric": np.asarray(results.electric),
            "ion_x_history": np.asarray(results.ion_x),
            "ion_v_history": np.asarray(results.ion_v),
            "total_energy": np.asarray(results.total_energy),
            "gauss_linf": np.asarray(results.gauss_linf),
        },
    )


def _read_run_checkpoint(path):
    metadata, arrays = load_checkpoint(path)
    if metadata.get("simulation") != "reflecting_wall_2d2v":
        raise ValueError("checkpoint belongs to a different simulation")
    config = WallConfig2D(**metadata["configuration"])
    electrons = _particle_from_state(
        arrays["electron_x"], arrays["electron_u"], m_e, q_e, metadata["electron_weight"]
    )
    ions = _particle_from_state(
        arrays["ion_x_state"], arrays["ion_u"], config.ion_mass, q_i, metadata["ion_weight"]
    )
    results = Results2D2V(
        x_grid=(np.arange(config.n_x) + 0.5) * config.dx,
        y_grid=(np.arange(config.n_y) + 0.5) * config.dy,
        t=list(arrays["times"]),
        n_e=list(arrays["n_e"]),
        n_i=list(arrays["n_i"]),
        electric=list(arrays["electric"]),
        ion_x=list(arrays["ion_x_history"]),
        ion_v=list(arrays["ion_v_history"]),
        total_energy=list(arrays["total_energy"]),
        gauss_linf=list(arrays["gauss_linf"]),
        configuration=asdict(config),
    )
    return config, int(metadata["step"]), electrons, ions, results


def run(config=None, *, stop_time=None, checkpoint_path=None, resume_from=None, **overrides):
    """Run, pause, or resume a configured 2D2V reflecting-wall case.

    Existing keyword overrides such as ``run(dt=0.005, particles_y=80)`` remain
    supported. A checkpoint records particle phase space, configuration, step,
    and diagnostic history, allowing an exactly reproducible continuation.
    """

    if resume_from is not None:
        if overrides:
            raise ValueError("configuration overrides cannot be applied while resuming")
        stored_config, start_step, electrons, ions, results = _read_run_checkpoint(
            resume_from
        )
        if config is not None and config != stored_config:
            raise ValueError("the supplied configuration does not match the checkpoint")
        config = stored_config
    else:
        config = WallConfig2D() if config is None else config
        if overrides:
            unknown = set(overrides) - set(asdict(config))
            if unknown:
                raise TypeError(f"Unknown run override(s): {sorted(unknown)}")
            config = replace(config, **overrides)
        electrons, ions = initialize_particles(config)
        start_step = 0
        results = Results2D2V(
            x_grid=(np.arange(config.n_x) + 0.5) * config.dx,
            y_grid=(np.arange(config.n_y) + 0.5) * config.dy,
            configuration=asdict(config),
        )

    n_e, n_i, _, electric, gauss = solve_field(
        electrons, ions, config.dx, config.dy, config.n_x, config.n_y
    )

    def save(time):
        results.t.append(time)
        results.n_e.append(n_e.copy())
        results.n_i.append(n_i.copy())
        results.electric.append(electric.copy())
        results.ion_x.append(ions.x.copy())
        results.ion_v.append(ions.v.copy())
        results.total_energy.append(
            total_energy(electrons, ions, electric, config.dx, config.dy)
        )
        results.gauss_linf.append(gauss)

    if start_step == 0 and not results.t:
        save(0.0)
    requested_time = config.t_max if stop_time is None else min(stop_time, config.t_max)
    target_step = min(config.steps, int(np.ceil(requested_time / config.dt)))
    if target_step < start_step:
        raise ValueError("stop_time precedes the checkpoint time")
    for step in range(start_step + 1, target_step + 1):
        electric_kick(
            electrons, electric, config.dx, config.dy, 0.5 * config.dt,
            config.n_x, config.n_y,
        )
        electric_kick(
            ions, electric, config.dx, config.dy, 0.5 * config.dt,
            config.n_x, config.n_y,
        )
        drift_and_bound(electrons, config.dt, config.length_x, config.length_y)
        drift_and_bound(ions, config.dt, config.length_x, config.length_y)
        n_e, n_i, _, electric, gauss = solve_field(
            electrons, ions, config.dx, config.dy, config.n_x, config.n_y
        )
        electric_kick(
            electrons, electric, config.dx, config.dy, 0.5 * config.dt,
            config.n_x, config.n_y,
        )
        electric_kick(
            ions, electric, config.dx, config.dy, 0.5 * config.dt,
            config.n_x, config.n_y,
        )
        if step % config.save_interval == 0 or step == target_step:
            save(step * config.dt)
    results.final_electron_x = electrons.x.copy()
    results.final_electron_v = electrons.v.copy()
    if checkpoint_path is not None:
        _write_run_checkpoint(
            Path(checkpoint_path), config, target_step, electrons, ions, results
        )
    return results


if __name__ == "__main__":
    output = run()
    energy = np.asarray(output.total_energy)
    print("snapshots:", len(output.t))
    print("maximum relative energy drift:", np.max(np.abs((energy-energy[0])/energy[0])))
    print("maximum Gauss residual:", max(output.gauss_linf))
