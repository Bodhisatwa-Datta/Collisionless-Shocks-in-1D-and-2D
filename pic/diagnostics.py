"""Reusable diagnostics for reflecting-wall shock experiments.

The functions in this module operate on saved arrays rather than solver
objects. This keeps the physical definitions independent of a particular run
script and makes the same analysis usable for 1D and transversely averaged 2D
data.
"""

from dataclasses import dataclass

import numpy as np


TINY = np.finfo(float).tiny


@dataclass(frozen=True)
class FrontFit:
    """Linear fit to a tracked shock front."""

    speed: float
    intercept: float
    r_squared: float
    points_used: int


@dataclass(frozen=True)
class SpeciesMoments:
    """One-dimensional moments of particles inside a spatial window."""

    count: int
    number_per_length: float
    bulk_velocity: float
    temperature: float
    selection: np.ndarray


@dataclass(frozen=True)
class ShockMetrics:
    """Quantities used to assess a collisionless-shock candidate."""

    front_positions: np.ndarray
    final_front: float
    front_fit: FrontFit
    downstream_window: tuple[float, float]
    upstream_window: tuple[float, float]
    downstream_density: float
    upstream_density: float
    compression_ratio: float
    downstream_ion_velocity: float
    upstream_ion_velocity: float
    downstream_ion_temperature: float
    upstream_ion_temperature: float
    downstream_electron_temperature: float
    upstream_electron_temperature: float
    downstream_mach: float
    upstream_mach: float
    mass_flux_mismatch: float
    ion_temperature_ratio: float
    reflected_ion_fraction: float


def density_profile(density: np.ndarray) -> np.ndarray:
    """Return a 1D profile, averaging all transverse density dimensions."""

    values = np.asarray(density, dtype=float)
    if values.ndim == 0:
        raise ValueError("density must contain a spatial dimension")
    if values.ndim == 1:
        return values
    return np.mean(values, axis=tuple(range(1, values.ndim)))


def smooth_profile(values: np.ndarray, width: int = 21) -> np.ndarray:
    """Smooth a profile with an odd-width moving average."""

    profile = np.asarray(values, dtype=float)
    if profile.ndim != 1:
        raise ValueError("smooth_profile expects a one-dimensional array")
    if profile.size < 3:
        return profile.copy()
    width = int(width)
    if width < 1:
        raise ValueError("smoothing width must be positive")
    width = min(width, profile.size if profile.size % 2 else profile.size - 1)
    if width % 2 == 0:
        width -= 1
    if width == 1:
        return profile.copy()
    return np.convolve(profile, np.ones(width) / width, mode="same")


def mean_in_window(x: np.ndarray, values: np.ndarray, window: tuple[float, float]) -> float:
    """Mean of a 1D profile over an inclusive spatial window."""

    coordinates = np.asarray(x, dtype=float)
    profile = np.asarray(values, dtype=float)
    if coordinates.shape != profile.shape:
        raise ValueError("x and values must have the same shape")
    low, high = window
    if high <= low:
        raise ValueError("window upper bound must exceed lower bound")
    mask = (coordinates >= low) & (coordinates <= high)
    if not np.any(mask):
        raise ValueError(f"window {window} contains no grid points")
    return float(np.mean(profile[mask]))


def locate_density_front(
    x: np.ndarray,
    density: np.ndarray,
    downstream_window: tuple[float, float],
    upstream_window: tuple[float, float],
    search_window: tuple[float, float],
    *,
    smoothing_width: int = 21,
    minimum_compression: float = 1.1,
) -> float:
    """Locate the midpoint of a wall-to-upstream density transition.

    ``NaN`` is returned when the downstream state has not reached the requested
    compression or no downward threshold crossing lies in ``search_window``.
    """

    coordinates = np.asarray(x, dtype=float)
    profile = smooth_profile(density_profile(density), smoothing_width)
    if coordinates.shape != profile.shape:
        raise ValueError("x and the reduced density profile must have the same shape")
    downstream = mean_in_window(coordinates, profile, downstream_window)
    upstream = mean_in_window(coordinates, profile, upstream_window)
    if downstream < minimum_compression * upstream:
        return float("nan")
    threshold = 0.5 * (upstream + downstream)
    low, high = search_window
    candidates = np.flatnonzero(
        (coordinates > low) & (coordinates < high) & (profile < threshold)
    )
    return float(coordinates[candidates[0]]) if candidates.size else float("nan")


def fit_front_trajectory(
    times: np.ndarray,
    positions: np.ndarray,
    *,
    start_fraction: float = 0.4,
    minimum_points: int = 4,
) -> FrontFit:
    """Fit constant front speed over the later part of a trajectory."""

    time = np.asarray(times, dtype=float)
    position = np.asarray(positions, dtype=float)
    if time.shape != position.shape or time.ndim != 1:
        raise ValueError("times and positions must be one-dimensional and equally sized")
    if not 0.0 <= start_fraction < 1.0:
        raise ValueError("start_fraction must lie in [0, 1)")
    selected = np.isfinite(position) & (time >= start_fraction * time[-1])
    count = int(np.count_nonzero(selected))
    if count < minimum_points:
        raise ValueError(f"front fit needs at least {minimum_points} valid points; found {count}")
    speed, intercept = np.polyfit(time[selected], position[selected], 1)
    prediction = speed * time[selected] + intercept
    residual = float(np.sum((position[selected] - prediction) ** 2))
    total = float(np.sum((position[selected] - np.mean(position[selected])) ** 2))
    r_squared = 1.0 - residual / max(total, TINY)
    return FrontFit(float(speed), float(intercept), float(r_squared), count)


def species_moments(
    position: np.ndarray,
    velocity: np.ndarray,
    window: tuple[float, float],
    mass: float,
    *,
    spatial_axis: int = 0,
    velocity_component: int = 0,
) -> SpeciesMoments:
    """Calculate particle moments in a spatial sampling window."""

    location = np.asarray(position, dtype=float)
    speeds = np.asarray(velocity, dtype=float)
    x_component = location if location.ndim == 1 else location[:, spatial_axis]
    v_component = speeds if speeds.ndim == 1 else speeds[:, velocity_component]
    if x_component.shape != v_component.shape:
        raise ValueError("position and velocity must describe the same particles")
    low, high = window
    if high <= low:
        raise ValueError("window upper bound must exceed lower bound")
    selection = (x_component >= low) & (x_component <= high)
    count = int(np.count_nonzero(selection))
    if count == 0:
        raise ValueError(f"window {window} contains no particles")
    selected_velocity = v_component[selection]
    return SpeciesMoments(
        count=count,
        number_per_length=count / (high - low),
        bulk_velocity=float(np.mean(selected_velocity)),
        temperature=float(mass * np.var(selected_velocity)),
        selection=selection,
    )


def relative_energy_drift(energy: np.ndarray) -> float:
    """Maximum energy change relative to the initial saved value."""

    values = np.asarray(energy, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("energy must be a non-empty one-dimensional array")
    return float(np.max(np.abs((values - values[0]) / max(abs(values[0]), TINY))))


def analyze_shock(
    *,
    x: np.ndarray,
    times: np.ndarray,
    density_history: np.ndarray,
    ion_position: np.ndarray,
    ion_velocity: np.ndarray,
    electron_position: np.ndarray,
    electron_velocity: np.ndarray,
    ion_mass: float,
    front_downstream_window: tuple[float, float],
    front_upstream_window: tuple[float, float],
    front_search_window: tuple[float, float],
    downstream_offsets: tuple[float, float] = (1.2, 0.4),
    upstream_offsets: tuple[float, float] = (0.8, 2.4),
    wall_guard: float = 0.2,
    smoothing_width: int = 21,
    fit_start_fraction: float = 0.4,
) -> ShockMetrics:
    """Calculate a consistent set of shock metrics from a saved state."""

    coordinates = np.asarray(x, dtype=float)
    time = np.asarray(times, dtype=float)
    histories = np.asarray(density_history, dtype=float)
    if histories.shape[0] != time.size:
        raise ValueError("density_history and times must contain the same number of snapshots")
    fronts = np.asarray(
        [
            locate_density_front(
                coordinates,
                density,
                front_downstream_window,
                front_upstream_window,
                front_search_window,
                smoothing_width=smoothing_width,
            )
            for density in histories
        ]
    )
    fit = fit_front_trajectory(time, fronts, start_fraction=fit_start_fraction)
    if not np.isfinite(fronts[-1]):
        raise ValueError("the final density snapshot has no detectable front")
    front = float(fronts[-1])
    downstream = (max(wall_guard, front - downstream_offsets[0]), front - downstream_offsets[1])
    upstream = (front + upstream_offsets[0], front + upstream_offsets[1])

    final_density = density_profile(histories[-1])
    n2 = mean_in_window(coordinates, final_density, downstream)
    n1 = mean_in_window(coordinates, final_density, upstream)
    ion_down = species_moments(ion_position, ion_velocity, downstream, ion_mass)
    ion_up = species_moments(ion_position, ion_velocity, upstream, ion_mass)
    electron_down = species_moments(electron_position, electron_velocity, downstream, 1.0)
    electron_up = species_moments(electron_position, electron_velocity, upstream, 1.0)

    u1 = ion_up.bulk_velocity - fit.speed
    u2 = ion_down.bulk_velocity - fit.speed
    cs1 = np.sqrt(max(electron_up.temperature + 3.0 * ion_up.temperature, 0.0) / ion_mass)
    cs2 = np.sqrt(max(electron_down.temperature + 3.0 * ion_down.temperature, 0.0) / ion_mass)
    flux1 = n1 * u1
    flux2 = n2 * u2
    flux_mismatch = abs(flux2 - flux1) / max(abs(flux1), abs(flux2), TINY)
    upstream_vx = np.asarray(ion_velocity)[ion_up.selection, 0]

    return ShockMetrics(
        front_positions=fronts,
        final_front=front,
        front_fit=fit,
        downstream_window=downstream,
        upstream_window=upstream,
        downstream_density=n2,
        upstream_density=n1,
        compression_ratio=n2 / max(n1, TINY),
        downstream_ion_velocity=ion_down.bulk_velocity,
        upstream_ion_velocity=ion_up.bulk_velocity,
        downstream_ion_temperature=ion_down.temperature,
        upstream_ion_temperature=ion_up.temperature,
        downstream_electron_temperature=electron_down.temperature,
        upstream_electron_temperature=electron_up.temperature,
        downstream_mach=abs(u2) / max(cs2, TINY),
        upstream_mach=abs(u1) / max(cs1, TINY),
        mass_flux_mismatch=float(flux_mismatch),
        ion_temperature_ratio=ion_down.temperature / max(ion_up.temperature, TINY),
        reflected_ion_fraction=float(np.mean(upstream_vx > 0.0)),
    )
