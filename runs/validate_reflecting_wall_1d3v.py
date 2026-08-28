"""Quantitative validation of the reflecting-wall 1D3V shock candidate."""

from pic.diagnostics import analyze_shock, relative_energy_drift
from runs.reflecting_wall_1d3v import ION_MASS, run


def main():
    results = run()
    metrics = analyze_shock(
        x=results.x_grid,
        times=results.t,
        density_history=results.n_i,
        ion_position=results.ion_x[-1],
        ion_velocity=results.ion_v[-1],
        electron_position=results.final_electron_x,
        electron_velocity=results.final_electron_v,
        ion_mass=ION_MASS,
        front_downstream_window=(0.4, 1.2),
        front_upstream_window=(8.0, 12.0),
        front_search_window=(0.5, 8.0),
        smoothing_width=31,
    )
    energy_drift = relative_energy_drift(results.total_energy)

    checks = {
        "front_moves_outward": metrics.front_fit.speed > 0.0,
        "front_motion_is_coherent": metrics.front_fit.r_squared > 0.9,
        "density_is_compressed": metrics.compression_ratio > 1.2,
        "upstream_is_supersonic": metrics.upstream_mach > 1.0,
        "downstream_is_subsonic": metrics.downstream_mach < 1.0,
        "mass_flux_is_steady": metrics.mass_flux_mismatch < 0.2,
        "ions_are_heated": metrics.ion_temperature_ratio > 1.5,
        "ions_are_reflected": metrics.reflected_ion_fraction > 0.01,
        "energy_is_controlled": energy_drift < 0.02,
    }
    confirmed = all(checks.values())

    print("STRICT REFLECTING-WALL SHOCK VALIDATION")
    print(f"confirmed_collisionless_shock={confirmed}")
    print(f"final_front_x={metrics.final_front:.8e}")
    print(f"shock_speed_over_c={metrics.front_fit.speed:.8e}")
    print(f"front_trajectory_r_squared={metrics.front_fit.r_squared:.8e}")
    print(f"front_fit_points={metrics.front_fit.points_used}")
    print(f"compression_ratio={metrics.compression_ratio:.8e}")
    print(f"upstream_mach={metrics.upstream_mach:.8e}")
    print(f"downstream_mach={metrics.downstream_mach:.8e}")
    print(f"mass_flux_relative_mismatch={metrics.mass_flux_mismatch:.8e}")
    print(f"ion_temperature_ratio={metrics.ion_temperature_ratio:.8e}")
    print(f"upstream_reflected_ion_fraction={metrics.reflected_ion_fraction:.8e}")
    print(f"maximum_relative_energy_drift={energy_drift:.8e}")
    for name, passed in checks.items():
        print(f"check_{name}={passed}")


if __name__ == "__main__":
    main()
