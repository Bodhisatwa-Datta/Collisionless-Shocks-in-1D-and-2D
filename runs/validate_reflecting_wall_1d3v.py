"""Strict physical validation of the reflecting-wall shock candidate."""

import sys

sys.path.append("./")
sys.path.append("../")

import numpy as np

from runs.reflecting_wall_1d3v import ION_MASS, run


def smooth(values, width=31):
    kernel = np.ones(width) / width
    return np.convolve(np.asarray(values, dtype=float), kernel, mode="same")


def front_from_density(x, density):
    """Midpoint of the monotonic wall-to-upstream density transition."""
    profile = smooth(density)
    downstream = (x >= 0.4) & (x <= 1.2)
    upstream = (x >= 8.0) & (x <= 12.0)
    n2 = float(np.mean(profile[downstream]))
    n1 = float(np.mean(profile[upstream]))
    if n2 < 1.1 * n1:
        return np.nan
    threshold = 0.5 * (n1 + n2)
    candidates = np.flatnonzero((x > 0.5) & (x < 8.0) & (profile < threshold))
    if not len(candidates):
        return np.nan
    return float(x[candidates[0]])


def linear_front_fit(times, positions):
    finite = np.isfinite(positions) & (times >= 0.4 * times[-1])
    slope, intercept = np.polyfit(times[finite], positions[finite], 1)
    fitted = slope * times[finite] + intercept
    residual = np.sum((positions[finite] - fitted) ** 2)
    total = np.sum((positions[finite] - np.mean(positions[finite])) ** 2)
    r_squared = 1.0 - residual / max(total, 1e-30)
    return float(slope), float(r_squared)


def species_moments(xp, vp, low, high, mass):
    selection = (xp >= low) & (xp <= high)
    velocity = vp[selection, 0]
    density_proxy = int(np.count_nonzero(selection)) / max(high - low, 1e-30)
    mean = float(np.mean(velocity))
    temperature = float(mass * np.var(velocity))
    return density_proxy, mean, temperature, selection


def main():
    results = run()
    x = results.x_grid
    times = np.asarray(results.t)
    fronts = np.asarray(
        [front_from_density(x, density) for density in results.n_i], dtype=float
    )
    shock_speed, trajectory_r2 = linear_front_fit(times, fronts)
    front = float(fronts[-1])

    downstream_limits = (max(0.2, front - 1.2), front - 0.4)
    upstream_limits = (front + 0.8, front + 2.4)
    ion_x = np.asarray(results.ion_x[-1])
    ion_v = np.asarray(results.ion_v[-1])
    electron_x = np.asarray(results.final_electron_x)
    electron_v = np.asarray(results.final_electron_v)

    ni2_proxy, vi2, ti2, ion_down = species_moments(
        ion_x, ion_v, *downstream_limits, ION_MASS
    )
    ni1_proxy, vi1, ti1, ion_up = species_moments(
        ion_x, ion_v, *upstream_limits, ION_MASS
    )
    _, _, te2, _ = species_moments(
        electron_x, electron_v, *downstream_limits, 1.0
    )
    _, _, te1, _ = species_moments(
        electron_x, electron_v, *upstream_limits, 1.0
    )

    # Shock-frame flow and 1D ion-acoustic estimates.
    u1 = vi1 - shock_speed
    u2 = vi2 - shock_speed
    cs1 = float(np.sqrt(max(te1 + 3.0 * ti1, 0.0) / ION_MASS))
    cs2 = float(np.sqrt(max(te2 + 3.0 * ti2, 0.0) / ION_MASS))
    mach1 = abs(u1) / max(cs1, 1e-30)
    mach2 = abs(u2) / max(cs2, 1e-30)

    density = np.asarray(results.n_i[-1])
    n2 = float(np.mean(density[(x >= downstream_limits[0]) & (x <= downstream_limits[1])]))
    n1 = float(np.mean(density[(x >= upstream_limits[0]) & (x <= upstream_limits[1])]))
    compression = n2 / max(n1, 1e-30)
    flux1 = n1 * u1
    flux2 = n2 * u2
    flux_mismatch = abs(flux2 - flux1) / max(abs(flux1), abs(flux2), 1e-30)
    reflected_fraction = float(np.mean(ion_v[ion_up, 0] > 0.0))
    heating = ti2 / max(ti1, 1e-30)
    energy = np.asarray(results.total_energy)
    energy_drift = float(np.max(np.abs((energy - energy[0]) / energy[0])))

    checks = {
        "front_moves_outward": shock_speed > 0.0,
        "front_motion_is_coherent": trajectory_r2 > 0.9,
        "density_is_compressed": compression > 1.2,
        "upstream_is_supersonic": mach1 > 1.0,
        "downstream_is_subsonic": mach2 < 1.0,
        "mass_flux_is_steady": flux_mismatch < 0.2,
        "ions_are_heated": heating > 1.5,
        "ions_are_reflected": reflected_fraction > 0.01,
        "energy_is_controlled": energy_drift < 0.02,
    }
    confirmed = all(checks.values())

    print("STRICT REFLECTING-WALL SHOCK VALIDATION")
    print(f"confirmed_collisionless_shock={confirmed}")
    print(f"final_front_x={front:.8e}")
    print(f"shock_speed_over_c={shock_speed:.8e}")
    print(f"front_trajectory_r_squared={trajectory_r2:.8e}")
    print(f"compression_ratio={compression:.8e}")
    print(f"upstream_mach={mach1:.8e}")
    print(f"downstream_mach={mach2:.8e}")
    print(f"mass_flux_relative_mismatch={flux_mismatch:.8e}")
    print(f"ion_temperature_ratio={heating:.8e}")
    print(f"upstream_reflected_ion_fraction={reflected_fraction:.8e}")
    print(f"maximum_relative_energy_drift={energy_drift:.8e}")
    for name, passed in checks.items():
        print(f"check_{name}={passed}")


if __name__ == "__main__":
    main()
