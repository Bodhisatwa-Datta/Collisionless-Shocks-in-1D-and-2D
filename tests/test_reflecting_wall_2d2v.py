import unittest
from pathlib import Path
import tempfile

import numpy as np

from runs.reflecting_wall_2d2v import (
    LENGTH_X,
    LENGTH_Y,
    N_X,
    N_Y,
    WallConfig2D,
    deposit_density,
    drift_and_bound,
    run,
    solve_field,
)
from pic.particles import Particles


class ReflectingWall2D2VTests(unittest.TestCase):
    def test_deposition_conserves_particle_number(self):
        particles = Particles(4, 2, 2, 1.0, -1.0)
        particles.x[:] = [[0.001, 0.001], [1.2, 2.1], [5.4, 3.999], [LENGTH_X-0.001, 1.0]]
        particles.weight = 0.3
        density = deposit_density(particles, LENGTH_X/N_X, LENGTH_Y/N_Y)
        integral = LENGTH_X/N_X * LENGTH_Y/N_Y * np.sum(density)
        self.assertAlmostEqual(integral, particles.N * particles.weight)

    def test_mixed_boundary_gauss_solve_is_accurate(self):
        electrons = Particles(3, 2, 2, 1.0, -1.0)
        ions = Particles(3, 2, 2, 25.0, 1.0)
        electrons.x[:] = [[0.4, 0.7], [2.1, 1.2], [3.0, 3.1]]
        ions.x[:] = [[0.6, 0.8], [2.4, 1.7], [3.4, 3.0]]
        electrons.weight = ions.weight = 0.5
        *_, residual = solve_field(electrons, ions, LENGTH_X/N_X, LENGTH_Y/N_Y)
        self.assertLess(residual, 1e-12)

    def test_x_reflection_and_y_periodicity_preserve_speed(self):
        particles = Particles(2, 2, 2, 1.0, 1.0)
        particles.x[:] = [[0.001, LENGTH_Y-0.001], [LENGTH_X-0.001, 0.001]]
        particles.v[:] = [[-0.2, 0.1], [0.2, -0.1]]
        particles.v_to_u()
        before = np.linalg.norm(particles.v, axis=1)
        drift_and_bound(particles, 0.1)
        self.assertTrue(np.all((particles.x[:,0] >= 0) & (particles.x[:,0] <= LENGTH_X)))
        self.assertTrue(np.all((particles.x[:,1] >= 0) & (particles.x[:,1] < LENGTH_Y)))
        np.testing.assert_allclose(np.linalg.norm(particles.v, axis=1), before)

    def test_checkpoint_resume_matches_uninterrupted_run(self):
        config = WallConfig2D(
            length_x=4.0,
            length_y=1.0,
            n_x=16,
            n_y=8,
            particles_x=20,
            particles_y=8,
            ion_mass=25.0,
            inflow_speed=-0.03,
            electron_thermal_speed=0.04,
            ion_thermal_speed=0.002,
            dt=0.02,
            t_max=0.2,
            save_interval=5,
            seed=23,
        )
        uninterrupted = run(config)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "wall-2d-state.npz"
            run(config, stop_time=0.1, checkpoint_path=checkpoint)
            resumed = run(resume_from=checkpoint)

        np.testing.assert_allclose(resumed.t, uninterrupted.t, rtol=0, atol=0)
        np.testing.assert_allclose(resumed.n_i, uninterrupted.n_i, rtol=0, atol=0)
        np.testing.assert_allclose(resumed.electric, uninterrupted.electric, rtol=0, atol=0)
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
