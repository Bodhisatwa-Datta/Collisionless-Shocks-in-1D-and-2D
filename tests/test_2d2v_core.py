import unittest

import numpy as np

from grids import Grid2D
from maxwell import (
    calc_charge_conserving_current_2D,
    calc_fields_2D,
    calc_gauss_residual_2D,
    initialize_electric_field_2D,
)
from newton import boris_pusher_2D2V
from particles import Particles


class Maxwell2D2VTests(unittest.TestCase):
    def test_vacuum_tmz_mode_has_bounded_energy(self):
        grid = Grid2D(1.0, 32)
        dt = grid.dx / (20 * np.sqrt(2))
        xx, yy = np.meshgrid(grid.x, grid.y, indexing="ij")
        grid.E[:, :, 1] = np.sin(2 * np.pi * 5 * xx) * np.sin(2 * np.pi * 3 * yy)
        initial = 0.5 * grid.dx**2 * (
            np.sum(grid.E**2) + np.sum(grid.B**2)
        )
        energies = []
        for _ in range(2_000):
            calc_fields_2D(grid, dt)
            energies.append(
                0.5 * grid.dx**2
                * (np.sum(grid.E**2) + np.sum(grid.B**2))
            )
        self.assertLess(max(energies), 1.01 * initial)
        self.assertGreater(min(energies), 0.99 * initial)


class Coupling2D2VTests(unittest.TestCase):
    @staticmethod
    def _species(positions, velocities, charge, mass=1.0, n0=1.0):
        positions = np.asarray(positions, dtype=float).reshape(-1, 2)
        velocities = np.asarray(velocities, dtype=float).reshape(-1, 2)
        particles = Particles(len(positions), 2, 2, mass, charge)
        particles.x[:] = positions
        particles.v[:] = velocities
        particles.v_to_u()
        particles.weight = n0 / particles.N
        return particles

    def test_ion_density_is_deposited_into_ion_array(self):
        electrons = self._species([[0.1, 0.1]], [[0, 0]], -1.0)
        ions = self._species([[0.7, 0.7]], [[0, 0]], 1.0, mass=1836.0)
        grid = Grid2D(1.0, 16)
        grid.set_densities(electrons, ions)
        self.assertAlmostEqual(grid.dx**2 * np.sum(grid.n_e), 1.0)
        self.assertAlmostEqual(grid.dx**2 * np.sum(grid.n_i), 1.0)
        self.assertFalse(np.allclose(grid.n_e, grid.n_i))

    def test_gauss_initialization_and_continuity_projection(self):
        dt = 0.04
        electrons = self._species([[0.98, 0.97]], [[0.4, 0.3]], -1.0)
        ions = self._species([[0.3, 0.4]], [[0, 0]], 1.0, mass=1836.0)
        grid = Grid2D(1.0, 24)
        grid.set_densities(electrons, ions)
        initialize_electric_field_2D(grid)
        self.assertLess(np.max(np.abs(calc_gauss_residual_2D(grid))), 1e-11)

        rho_old = grid.rho.copy()
        electron_old = electrons.x.copy()
        ion_old = ions.x.copy()
        electrons.x[:] = (electrons.x + electrons.v * dt) % 1.0
        ions.x[:] = (ions.x + ions.v * dt) % 1.0
        grid.set_densities(electrons, ions)
        calc_charge_conserving_current_2D(
            grid, electrons, ions, electron_old, ion_old, rho_old, dt
        )
        self.assertLess(np.max(np.abs(grid.continuity_residual)), 1e-10)
        calc_fields_2D(grid, dt)
        self.assertLess(np.max(np.abs(calc_gauss_residual_2D(grid))), 1e-10)

    def test_uniform_b_preserves_relativistic_speed(self):
        particle = self._species([[0.3, 0.4]], [[0.3, 0.4]], -1.0)
        grid = Grid2D(1.0, 16)
        grid.B[:, :, 0] = 1.0
        initial_u = np.linalg.norm(particle.u)
        for _ in range(1_000):
            boris_pusher_2D2V(grid, particle, 0.01)
        self.assertAlmostEqual(np.linalg.norm(particle.u), initial_u, places=12)
        self.assertAlmostEqual(np.linalg.norm(particle.v), 0.5, places=12)

    def test_electron_plasma_oscillation_has_normalized_frequency(self):
        particles_per_side = 20
        n_cells = 16
        dt = (1.0 / n_cells) / (20 * np.sqrt(2))
        coordinates = (np.arange(particles_per_side) + 0.5) / particles_per_side
        xx, yy = np.meshgrid(coordinates, coordinates, indexing="ij")
        positions = np.column_stack((xx.ravel(), yy.ravel()))
        electron_positions = positions.copy()
        electron_positions[:, 0] += 1e-3 * np.cos(2 * np.pi * electron_positions[:, 0])
        electron_positions %= 1.0
        velocities = np.zeros_like(positions)
        electrons = self._species(electron_positions, velocities, -1.0)
        ions = self._species(positions, velocities, 1.0, mass=1836.0)
        grid = Grid2D(1.0, n_cells)
        grid.set_densities(electrons, ions)
        initialize_electric_field_2D(grid)
        initial_mode = grid.E[:, :, 0].copy()
        boris_pusher_2D2V(grid, electrons, dt / 2)
        boris_pusher_2D2V(grid, ions, dt / 2)

        previous_amplitude = np.sum(grid.E[:, :, 0] * initial_mode)
        previous_time = 0.0
        zero_crossing = None
        for step in range(1, 900):
            rho_old = grid.rho.copy()
            electron_old = electrons.x.copy()
            ion_old = ions.x.copy()
            electrons.x[:] = (electrons.x + electrons.v * dt) % 1.0
            ions.x[:] = (ions.x + ions.v * dt) % 1.0
            grid.set_densities(electrons, ions)
            calc_charge_conserving_current_2D(
                grid, electrons, ions, electron_old, ion_old, rho_old, dt
            )
            calc_fields_2D(grid, dt)
            amplitude = np.sum(grid.E[:, :, 0] * initial_mode)
            current_time = step * dt
            if zero_crossing is None and previous_amplitude * amplitude <= 0:
                zero_crossing = previous_time - previous_amplitude * (
                    current_time - previous_time
                ) / (amplitude - previous_amplitude)
            previous_amplitude = amplitude
            previous_time = current_time
            boris_pusher_2D2V(grid, electrons, dt)
            boris_pusher_2D2V(grid, ions, dt)

        self.assertIsNotNone(zero_crossing)
        measured_frequency = np.pi / (2 * zero_crossing)
        self.assertAlmostEqual(measured_frequency, 1.0, delta=0.03)
        self.assertLess(
            np.max(np.abs(calc_gauss_residual_2D(grid))), 1e-11
        )


if __name__ == "__main__":
    unittest.main()
