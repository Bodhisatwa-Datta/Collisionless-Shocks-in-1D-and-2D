import unittest

import numpy as np

from runs.reflecting_wall_2d2v import (
    LENGTH_X,
    LENGTH_Y,
    N_X,
    N_Y,
    deposit_density,
    drift_and_bound,
    solve_field,
)
from particles import Particles


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


if __name__ == "__main__":
    unittest.main()
