import numpy as np
import numpy.typing as npt

from .constants import c


class Particles:
    def __init__(self, num_particles: int, dimX: int, dimV: int, mass: float, charge: float):
        self.x = np.empty((num_particles, dimX))
        self.v = np.empty((num_particles, dimV))
        self.u = np.empty((num_particles, dimV))
        self.idx: npt.NDArray = np.empty((num_particles, dimX), dtype=np.int32)
        self.idx_staggered: npt.NDArray = np.empty((num_particles, dimX), dtype=np.int32)
        # One linear CIC weight per spatial coordinate.
        self.cic_weights: npt.NDArray = np.empty((num_particles, dimX), dtype=np.float64)
        self.cic_weights_staggered: npt.NDArray = np.empty((num_particles, dimX), dtype=np.float64)
        self.m = mass
        self.q = charge
        self.qm = charge / mass
        # Number of physical particles represented by one macro-particle.
        # The 1D3V solver sets this from n0 * domain_length / N_species.
        self.weight = 1.0
        self.N = num_particles
        self.dimX = dimX
        self.dimV = dimV

    def filter(self, mask: npt.NDArray):
        """
        Only keep the particles where mask[i] = True.
        mask should be a 1D boolean array of the same length as x and y
        """
        self.x = self.x[mask]
        self.v = self.v[mask]
        self.N = self.x.shape[0]

    def add_particles(self, x_new: npt.NDArray, v_new: npt.NDArray):
        """
        Add new particles with given speed and velocity
        """
        self.x = np.concatenate((self.x, x_new))
        self.v = np.concatenate((self.v, v_new))
        self.N = self.x.shape[0]

    def kinetic_energy(self):
        """
        Return the total kinetic energy of the particles
        KE = 1/2 * m * sum_{j=1}^{N} v_j**2
        """
        return 0.5 * self.m * np.sum(self.v**2)

    def relativistic_kinetic_energy(self):
        """Return sum_p w_p m c^2 (gamma_p - 1)."""
        gamma = np.sqrt(1 + np.sum(self.u**2, axis=1) / (c * c))
        return self.weight * self.m * c * c * np.sum(gamma - 1)

    def v_to_u(self):
        speed_sq = np.sum(self.v**2, axis=1, keepdims=True)
        if np.any(speed_sq >= c * c):
            raise ValueError("Particle speed must be smaller than the speed of light")
        gamma = 1 / np.sqrt(1 - speed_sq / (c * c))
        self.u = gamma * self.v
