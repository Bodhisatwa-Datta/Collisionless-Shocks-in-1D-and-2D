import unittest

import numpy as np

from pic.fields import (
    calc_charge_conserving_current_1D3V,
    calc_fields_1D3V,
    calc_gauss_residual_1D3V,
    initialize_electric_field_1D3V,
)
from pic.grids import Grid1D3V
from pic.particles import Particles
from pic.pushers import boris_pusher_1D3V


class Maxwell1D3VTests(unittest.TestCase):
    def test_vacuum_transverse_mode_has_bounded_energy(self):
        n_cells = 200
        grid = Grid1D3V(1.0, n_cells)
        dt = grid.dx / 20
        mode = 50
        grid.E[:, 1] = np.sin(2 * np.pi * mode * grid.x)

        initial_energy = 0.5 * grid.dx * np.sum(grid.E**2 + grid.B**2)
        energies = []
        for _ in range(2_000):
            calc_fields_1D3V(grid, dt)
            energies.append(0.5 * grid.dx * np.sum(grid.E**2 + grid.B**2))

        # A Yee/Verlet wave has a small bounded energy oscillation. The old
        # composition grew this mode exponentially by orders of magnitude.
        self.assertLess(max(energies), 1.01 * initial_energy)
        self.assertGreater(min(energies), 0.99 * initial_energy)


class ParticleCouplingTests(unittest.TestCase):
    @staticmethod
    def _particle(position, velocity, charge=-1.0, mass=1.0):
        particle = Particles(1, 1, 3, mass, charge)
        particle.x[:] = position
        particle.v[:] = velocity
        particle.v_to_u()
        return particle

    def test_staggered_magnetic_cic_weight_uses_half_cell_origin(self):
        particle = self._particle(0.37, [0.0, 0.0, 0.0])
        other = self._particle(0.37, [0.0, 0.0, 0.0], charge=1.0)
        grid = Grid1D3V(1.0, 10)
        grid.set_densities(particle, other)

        self.assertEqual(particle.idx_staggered.item(), 3)
        self.assertAlmostEqual(particle.cic_weights_staggered.item(), 0.2)

        # B[j] represents x=(j+1/2)dx. Linear interpolation must recover x.
        grid.B[:, 0] = (np.arange(grid.n_cells) + 0.5) * grid.dx
        j = particle.idx_staggered.item()
        weight = particle.cic_weights_staggered.item()
        interpolated = grid.B[j, 0] * (1 - weight) + grid.B[j + 1, 0] * weight
        self.assertAlmostEqual(interpolated, particle.x.item())

    def test_relativistic_boris_preserves_speed_in_uniform_magnetic_field(self):
        particle = self._particle(0.25, [0.0, 0.6, 0.0])
        other = self._particle(0.25, [0.0, 0.0, 0.0], charge=1.0)
        grid = Grid1D3V(1.0, 10)
        grid.set_densities(particle, other)
        grid.B[:, 0] = 1.0

        initial_u_norm = np.linalg.norm(particle.u)
        speeds = []
        for _ in range(1_000):
            boris_pusher_1D3V(grid, particle, 0.01)
            speeds.append(np.linalg.norm(particle.v))

        self.assertAlmostEqual(np.linalg.norm(particle.u), initial_u_norm, places=12)
        self.assertLess(max(abs(np.asarray(speeds) - 0.6)), 1e-12)

    def test_velocity_to_momentum_uses_total_speed(self):
        particle = self._particle(0.25, [0.3, 0.4, 0.0])
        gamma = 1 / np.sqrt(1 - 0.5**2)
        np.testing.assert_allclose(particle.u, gamma * particle.v)


class ConstraintTests(unittest.TestCase):
    @staticmethod
    def _species(positions, velocities, charge, mass=1.0, n0=1.0):
        positions = np.asarray(positions, dtype=float).reshape(-1, 1)
        velocities = np.asarray(velocities, dtype=float).reshape(-1, 3)
        particles = Particles(len(positions), 1, 3, mass, charge)
        particles.x[:] = positions
        particles.v[:] = velocities
        particles.v_to_u()
        particles.weight = n0 / particles.N
        return particles

    def test_macro_particle_count_does_not_change_mean_density(self):
        for particle_count in (32, 128):
            positions = np.arange(particle_count) / particle_count
            velocities = np.zeros((particle_count, 3))
            electrons = self._species(positions, velocities, -1.0)
            ions = self._species(positions, velocities, 1.0, mass=1836.0)
            grid = Grid1D3V(1.0, 32)
            grid.set_densities(electrons, ions)

            self.assertAlmostEqual(grid.dx * np.sum(grid.n_e), 1.0)
            self.assertAlmostEqual(grid.dx * np.sum(grid.n_i), 1.0)
            self.assertAlmostEqual(np.mean(grid.n_e), 1.0)
            self.assertLess(np.max(np.abs(grid.rho)), 1e-14)

    def test_continuity_deposition_preserves_gauss_law(self):
        dt = 0.1
        electrons = self._species([[0.99]], [[0.3, 0.2, -0.1]], -1.0)
        ions = self._species([[0.35]], [[0.0, 0.0, 0.0]], 1.0, mass=1836.0)
        grid = Grid1D3V(1.0, 32)
        grid.set_densities(electrons, ions)
        initialize_electric_field_1D3V(grid)

        rho_old = grid.rho.copy()
        electron_x_old = electrons.x.copy()
        ion_x_old = ions.x.copy()
        electrons.x[:, 0] = (electrons.x[:, 0] + electrons.v[:, 0] * dt) % 1.0
        ions.x[:, 0] = (ions.x[:, 0] + ions.v[:, 0] * dt) % 1.0

        grid.set_densities(electrons, ions)
        calc_charge_conserving_current_1D3V(
            grid,
            electrons,
            ions,
            electron_x_old,
            ion_x_old,
            rho_old,
            dt,
        )
        self.assertLess(np.max(np.abs(grid.continuity_residual)), 1e-12)

        calc_fields_1D3V(grid, dt)
        residual = calc_gauss_residual_1D3V(grid)
        self.assertLess(np.max(np.abs(residual)), 1e-11)

    def test_electron_plasma_oscillation_has_normalized_frequency(self):
        particle_count = 320
        n_cells = 32
        dt = (1.0 / n_cells) / 20
        base_positions = np.arange(particle_count) / particle_count

        electrons = self._species(
            (base_positions + 1e-3 * np.cos(2 * np.pi * base_positions)) % 1.0,
            np.zeros((particle_count, 3)),
            -1.0,
        )
        ions = self._species(
            base_positions,
            np.zeros((particle_count, 3)),
            1.0,
            mass=1836.0,
        )
        grid = Grid1D3V(1.0, n_cells)
        grid.set_densities(electrons, ions)
        initialize_electric_field_1D3V(grid)
        initial_mode = grid.E[:, 0].copy()
        boris_pusher_1D3V(grid, electrons, dt / 2)
        boris_pusher_1D3V(grid, ions, dt / 2)

        previous_amplitude = np.dot(grid.E[:, 0], initial_mode)
        previous_time = 0.0
        zero_crossing = None
        max_gauss_error = 0.0
        max_transverse_field = 0.0

        for step in range(1, 1_400):
            rho_old = grid.rho.copy()
            electron_x_old = electrons.x.copy()
            ion_x_old = ions.x.copy()
            electrons.x[:, 0] = (electrons.x[:, 0] + electrons.v[:, 0] * dt) % 1.0
            ions.x[:, 0] = (ions.x[:, 0] + ions.v[:, 0] * dt) % 1.0
            grid.set_densities(electrons, ions)
            calc_charge_conserving_current_1D3V(
                grid,
                electrons,
                ions,
                electron_x_old,
                ion_x_old,
                rho_old,
                dt,
            )
            calc_fields_1D3V(grid, dt)

            amplitude = np.dot(grid.E[:, 0], initial_mode)
            current_time = step * dt
            if zero_crossing is None and previous_amplitude * amplitude <= 0:
                zero_crossing = previous_time - previous_amplitude * (
                    current_time - previous_time
                ) / (amplitude - previous_amplitude)
            previous_amplitude = amplitude
            previous_time = current_time

            max_gauss_error = max(
                max_gauss_error,
                np.max(np.abs(calc_gauss_residual_1D3V(grid))),
            )
            max_transverse_field = max(
                max_transverse_field,
                np.max(np.abs(grid.E[:, 1:])),
                np.max(np.abs(grid.B[:, 1:])),
            )
            boris_pusher_1D3V(grid, electrons, dt)
            boris_pusher_1D3V(grid, ions, dt)

        self.assertIsNotNone(zero_crossing)
        measured_frequency = np.pi / (2 * zero_crossing)
        self.assertAlmostEqual(measured_frequency, 1.0, delta=0.03)
        self.assertLess(max_gauss_error, 1e-11)
        self.assertEqual(max_transverse_field, 0.0)


if __name__ == "__main__":
    unittest.main()
