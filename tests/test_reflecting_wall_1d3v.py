import unittest
from pathlib import Path
import tempfile

import numpy as np

from pic.particles import Particles
from runs.reflecting_wall_1d3v import (
    LENGTH,
    N_CELLS,
    WallConfig,
    deposit_number_density,
    drift_and_reflect,
    run,
    solve_field,
)


class ReflectingWallTests(unittest.TestCase):
    def test_nonperiodic_deposition_conserves_particle_number(self):
        particles = Particles(4, 1, 3, 1.0, -1.0)
        particles.x[:, 0] = [0.001, 0.75, LENGTH - 0.75, LENGTH - 0.001]
        particles.weight = 0.4
        dx = LENGTH / N_CELLS
        density = deposit_number_density(particles, dx, N_CELLS)
        self.assertAlmostEqual(dx * np.sum(density), particles.N * particles.weight)

    def test_nonperiodic_gauss_solve_is_exact(self):
        electrons = Particles(2, 1, 3, 1.0, -1.0)
        ions = Particles(2, 1, 3, 25.0, 1.0)
        electrons.x[:, 0] = [0.31, 1.27]
        ions.x[:, 0] = [0.42, 1.51]
        electrons.weight = ions.weight = 0.5
        dx = LENGTH / N_CELLS
        _, _, _, _, residual = solve_field(electrons, ions, dx, N_CELLS)
        self.assertLess(residual, 1e-13)

    def test_specular_wall_preserves_speed(self):
        particles = Particles(2, 1, 3, 1.0, 1.0)
        particles.x[:, 0] = [0.001, LENGTH - 0.001]
        particles.v[:] = [[-0.2, 0.03, 0.0], [0.2, -0.04, 0.0]]
        particles.v_to_u()
        speed_before = np.linalg.norm(particles.v, axis=1)
        drift_and_reflect(particles, 0.1)
        self.assertTrue(np.all(particles.x[:, 0] >= 0.0))
        self.assertTrue(np.all(particles.x[:, 0] <= LENGTH))
        np.testing.assert_allclose(np.linalg.norm(particles.v, axis=1), speed_before)

    def test_checkpoint_resume_matches_uninterrupted_run(self):
        config = WallConfig(
            length=4.0,
            n_cells=32,
            particles_per_species=256,
            ion_mass=25.0,
            inflow_speed=-0.03,
            electron_thermal_speed=0.04,
            ion_thermal_speed=0.002,
            dt=0.02,
            t_max=0.4,
            save_interval=5,
            seed=19,
        )
        uninterrupted = run(config)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "wall-state.npz"
            run(config, stop_time=0.2, checkpoint_path=checkpoint)
            resumed = run(resume_from=checkpoint)

        np.testing.assert_allclose(resumed.t, uninterrupted.t, rtol=0, atol=0)
        np.testing.assert_allclose(resumed.n_i, uninterrupted.n_i, rtol=0, atol=0)
        np.testing.assert_allclose(resumed.ex, uninterrupted.ex, rtol=0, atol=0)
        np.testing.assert_allclose(
            resumed.ion_x[-1], uninterrupted.ion_x[-1], rtol=0, atol=0
        )
        np.testing.assert_allclose(
            resumed.ion_v[-1], uninterrupted.ion_v[-1], rtol=0, atol=0
        )
        np.testing.assert_allclose(
            resumed.total_energy, uninterrupted.total_energy, rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
