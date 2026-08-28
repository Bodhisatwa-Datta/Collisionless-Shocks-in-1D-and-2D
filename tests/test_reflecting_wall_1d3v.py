import unittest

import numpy as np

from pic.particles import Particles
from runs.reflecting_wall_1d3v import (
    LENGTH,
    N_CELLS,
    deposit_number_density,
    drift_and_reflect,
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


if __name__ == "__main__":
    unittest.main()
