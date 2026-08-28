import unittest

import numpy as np

from pic.diagnostics import (
    analyze_shock,
    density_profile,
    fit_front_trajectory,
    locate_density_front,
    relative_energy_drift,
    species_moments,
)


class ShockDiagnosticTests(unittest.TestCase):
    def test_transverse_density_is_reduced_to_x_profile(self):
        density = np.arange(24, dtype=float).reshape(4, 3, 2)
        np.testing.assert_allclose(density_profile(density), np.mean(density, axis=(1, 2)))

    def test_density_front_tracks_known_step(self):
        x = np.linspace(0.0, 10.0, 501)
        density = np.where(x < 3.0, 2.0, 1.0)
        front = locate_density_front(
            x,
            density,
            downstream_window=(0.5, 1.5),
            upstream_window=(7.0, 9.0),
            search_window=(1.5, 6.0),
            smoothing_width=9,
        )
        self.assertAlmostEqual(front, 3.0, delta=0.06)

    def test_front_fit_ignores_early_and_missing_positions(self):
        times = np.linspace(0.0, 10.0, 21)
        positions = 1.2 + 0.15 * times
        positions[:5] = np.nan
        fit = fit_front_trajectory(times, positions, start_fraction=0.4)
        self.assertAlmostEqual(fit.speed, 0.15, places=12)
        self.assertAlmostEqual(fit.r_squared, 1.0, places=12)
        self.assertEqual(fit.points_used, 13)

    def test_particle_moments_use_requested_window(self):
        position = np.array([[0.2, 4.0], [0.6, 5.0], [0.8, 6.0]])
        velocity = np.array([[9.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
        moments = species_moments(position, velocity, (0.5, 1.0), mass=2.0)
        self.assertEqual(moments.count, 2)
        self.assertAlmostEqual(moments.bulk_velocity, 2.0)
        self.assertAlmostEqual(moments.temperature, 2.0)

    def test_shared_analysis_handles_transversely_uniform_2d_density(self):
        rng = np.random.default_rng(7)
        x = np.linspace(0.0, 10.0, 501)
        times = np.linspace(0.0, 10.0, 21)
        exact_fronts = 1.5 + 0.1 * times
        history_1d = np.asarray([np.where(x < front, 2.0, 1.0) for front in exact_fronts])
        history_2d = np.repeat(history_1d[:, :, None], 6, axis=2)

        final_front = exact_fronts[-1]
        down_x = rng.uniform(final_front - 1.15, final_front - 0.45, 300)
        up_x = rng.uniform(final_front + 0.85, final_front + 2.35, 300)
        positions = np.column_stack((np.concatenate((down_x, up_x)), rng.uniform(0, 1, 600)))
        ion_vx = np.concatenate(
            (rng.normal(-0.02, 0.015, 300), rng.normal(-0.12, 0.01, 300))
        )
        ion_vx[-20:] = 0.03
        ion_velocity = np.column_stack((ion_vx, rng.normal(0, 0.01, 600)))
        electron_velocity = np.column_stack(
            (rng.normal(-0.07, 0.03, 600), rng.normal(0, 0.03, 600))
        )

        metrics = analyze_shock(
            x=x,
            times=times,
            density_history=history_2d,
            ion_position=positions,
            ion_velocity=ion_velocity,
            electron_position=positions,
            electron_velocity=electron_velocity,
            ion_mass=100.0,
            front_downstream_window=(0.3, 0.9),
            front_upstream_window=(7.0, 9.0),
            front_search_window=(0.9, 6.0),
            smoothing_width=9,
        )
        self.assertAlmostEqual(metrics.front_fit.speed, 0.1, delta=0.005)
        self.assertAlmostEqual(metrics.compression_ratio, 2.0, places=12)
        self.assertGreater(metrics.reflected_ion_fraction, 0.01)

    def test_relative_energy_drift_uses_initial_energy(self):
        self.assertAlmostEqual(relative_energy_drift([10.0, 10.2, 9.9]), 0.02)


if __name__ == "__main__":
    unittest.main()
