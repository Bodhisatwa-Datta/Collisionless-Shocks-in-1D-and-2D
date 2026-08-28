import numpy as np

from .particles import Particles


# 1 spatial index, 1 component
class Grid1D:
    def __init__(self, x_max, n_cells):
        self.x_max = x_max
        self.n_cells = n_cells
        self.x = np.linspace(0, x_max, n_cells, endpoint=False)
        self.dx = self.x[1] - self.x[0]
        # Cell averaged-quantities
        self.E = np.zeros(self.n_cells)
        self.n_e = np.empty(self.n_cells)
        self.n_i = np.empty(self.n_cells)
        self.rho = np.empty(self.n_cells)
        self.phi = np.empty(self.n_cells)

    def set_densities(self, electrons: Particles, ions: Particles):
        """
        Set the electron density, ion density and charge density on the grid
        """
        # Density deposition also records the interpolation weights reused by
        # the particle pusher.
        dummy = electrons.x / self.dx
        np.floor(dummy, out=electrons.idx, casting="unsafe")
        electrons.cic_weights = dummy - electrons.idx
        self.n_e.fill(0)
        np.add.at(self.n_e, electrons.idx, 1 - electrons.cic_weights)
        # Grid1D is the legacy periodic electrostatic grid.
        np.add.at(self.n_e, (electrons.idx + 1) % self.n_cells, electrons.cic_weights)

        dummy = ions.x / self.dx
        np.floor(dummy, out=ions.idx, casting="unsafe")
        ions.cic_weights = dummy - ions.idx
        self.n_i.fill(0)
        np.add.at(self.n_i, ions.idx, 1 - ions.cic_weights)
        np.add.at(self.n_i, (ions.idx + 1) % self.n_cells, ions.cic_weights)
        self.rho = electrons.q * self.n_e + ions.q * self.n_i


# 1 spatial index, 3 components, adding B and J
class Grid1D3V:
    def __init__(self, x_max, n_cells):
        self.x_max = x_max
        self.n_cells = n_cells
        self.x = np.linspace(0, x_max, n_cells, endpoint=False)
        self.dx = self.x[1] - self.x[0]
        # External fields:
        self.E_0 = np.zeros((self.n_cells, 3))
        self.B_0 = np.zeros((self.n_cells, 3))
        # Total fields
        self.E = np.zeros((self.n_cells, 3))
        # the x component is always calculated directly using the poisson solver,
        # so it has to be added there each timestep
        self.E[:, 1:] = self.E_0[:, 1:]
        self.J = np.zeros((self.n_cells, 3))
        self.B = np.zeros((self.n_cells, 3)) + self.B_0

        # Cell averaged-quantities
        self.n_e = np.empty(self.n_cells)
        self.n_i = np.empty(self.n_cells)
        self.rho = np.empty(self.n_cells)
        self.gauss_residual = np.zeros(self.n_cells)
        self.continuity_residual = np.zeros(self.n_cells)

    def set_densities(self, electrons: Particles, ions: Particles):
        """
        Set the electron density, ion density and charge density on the grid
        """
        dummy = electrons.x / self.dx
        np.floor(dummy, out=electrons.idx, casting="unsafe")
        np.floor(dummy - 0.5, out=electrons.idx_staggered, casting="unsafe")
        electrons.cic_weights = dummy - electrons.idx
        # B[j] lives at (j + 1/2) * dx, so measure the CIC distance
        # from that staggered location rather than from the integer E grid.
        electrons.cic_weights_staggered = dummy - (electrons.idx_staggered + 0.5)
        self.n_e.fill(0)
        np.add.at(
            self.n_e,
            electrons.idx,
            electrons.weight / self.dx * (1 - electrons.cic_weights),
        )
        # This electromagnetic grid is validated for periodic boundaries.
        np.add.at(
            self.n_e,
            (electrons.idx + 1) % self.n_cells,
            electrons.weight / self.dx * electrons.cic_weights,
        )

        dummy = ions.x / self.dx
        np.floor(dummy, out=ions.idx, casting="unsafe")
        np.floor(dummy - 0.5, out=ions.idx_staggered, casting="unsafe")
        ions.cic_weights = dummy - ions.idx
        ions.cic_weights_staggered = dummy - (ions.idx_staggered + 0.5)
        self.n_i.fill(0)
        np.add.at(
            self.n_i,
            ions.idx,
            ions.weight / self.dx * (1 - ions.cic_weights),
        )
        np.add.at(
            self.n_i,
            (ions.idx + 1) % self.n_cells,
            ions.weight / self.dx * ions.cic_weights,
        )
        self.rho[:] = electrons.q * self.n_e + ions.q * self.n_i


# 2 spatial indices and 2 velocity components (TMz electromagnetic system).
class Grid2D:
    """Square periodic Yee grid.

    rho lives at (i,j), Ex/Jx at (i+1/2,j), Ey/Jy at (i,j+1/2), and
    Bz at (i+1/2,j+1/2). All arrays contain exactly n_cells periodic samples
    per direction; there is no duplicated endpoint.
    """

    def __init__(self, x_max, n_cells):
        self.x_max = x_max
        self.n_cells = n_cells
        self.dx = x_max / n_cells
        self.x = np.linspace(0, x_max, n_cells, endpoint=False)
        self.y = np.linspace(0, x_max, n_cells, endpoint=False)
        self.E = np.zeros((n_cells, n_cells, 2))
        self.J = np.zeros((n_cells, n_cells, 2))
        self.B = np.zeros((n_cells, n_cells, 1))
        self.n_e = np.empty((n_cells, n_cells))
        self.n_i = np.empty((n_cells, n_cells))
        self.rho = np.empty((n_cells, n_cells))
        self.gauss_residual = np.zeros((n_cells, n_cells))
        self.continuity_residual = np.zeros((n_cells, n_cells))

    def _deposit_scalar(self, target, particles):
        scaled = particles.x / self.dx
        np.floor(scaled, out=particles.idx, casting="unsafe")
        particles.cic_weights = scaled - particles.idx
        ix = particles.idx[:, 0]
        iy = particles.idx[:, 1]
        wx = particles.cic_weights[:, 0]
        wy = particles.cic_weights[:, 1]
        scale = particles.weight / (self.dx * self.dx)
        np.add.at(target, (ix, iy), scale * (1 - wx) * (1 - wy))
        np.add.at(target, ((ix + 1) % self.n_cells, iy), scale * wx * (1 - wy))
        np.add.at(target, (ix, (iy + 1) % self.n_cells), scale * (1 - wx) * wy)
        np.add.at(
            target,
            ((ix + 1) % self.n_cells, (iy + 1) % self.n_cells),
            scale * wx * wy,
        )

    def set_densities(self, electrons: Particles, ions: Particles):
        self.n_e.fill(0)
        self.n_i.fill(0)
        self._deposit_scalar(self.n_e, electrons)
        self._deposit_scalar(self.n_i, ions)
        self.rho[:] = electrons.q * self.n_e + ions.q * self.n_i
