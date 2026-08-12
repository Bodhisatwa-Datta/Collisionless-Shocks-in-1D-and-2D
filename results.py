import os
import pickle
from typing import Union

import numpy as np

import grids
from parameters import Parameters
from particles import Particles
from physical_constants import eps_0, mu_0


class Results:
    # NOTE: calling .append() is not super cheap, due to re-/deallocation of memory.
    #   However, I think that python internally doubles the array-length at each reallocation,
    #   which is definitely better than increasing by one on each .append() call.
    #   If we think about it, we might be able to remove some .copy() calls if arrays are replaced in the next iteration anyway
    #   (e.g. we do not have to copy grid.E if in the next iteration we do grid.E = "some new array",
    #    since this statement does not modify the previous array, but makes E point to "some new array")

    def __init__(self):
        # Keeps track of the time at which snapshots were made
        self.t = []
        # Keeps track of the different types of energy
        self.KE = []
        self.PE = []
        self.TE = []
        # Optionally, we could store the charge density as well, this is not done currently but is easily added
        self.n_e = []
        self.n_i = []
        # Keeps track of phase-space (x,v) of electrons and ions
        self.x_e = []
        self.v_e = []
        self.x_i = []
        self.v_i = []
        # Keeps track of the electric field E(x)
        self.E = []
        # Keeps track of how cells shape the domain
        self.x = []

    def save(
        self,
        t: float,
        KE: float,
        PE: float,
        TE: float,
        grid: Union[grids.Grid1D, grids.Grid1D3V, grids.Grid2D],
        electrons: Particles,
        ions: Particles,
    ):
        self.save_time(t)
        self.save_energies(KE, PE, TE)
        self.save_densities(grid)
        self.save_phase_space(electrons, ions)
        self.save_fields(grid)
        self.save_cells(grid)

    def save_time(self, t: float):
        self.t.append(t)

    def save_energies(self, KE: float, PE: float, TE: float):
        self.KE.append(KE)
        self.PE.append(PE)
        self.TE.append(TE)

    def save_densities(self, grid: Union[grids.Grid1D, grids.Grid1D3V, grids.Grid2D]):
        self.n_e.append(grid.n_e.copy())
        self.n_i.append(grid.n_i.copy())

    def save_phase_space(self, electrons: Particles, ions: Particles):
        self.x_e.append(electrons.x.copy())
        self.v_e.append(electrons.v.copy())
        self.x_i.append(ions.x.copy())
        self.v_i.append(ions.v.copy())

    def save_fields(self, grid: Union[grids.Grid1D, grids.Grid1D3V, grids.Grid2D]):
        self.E.append(grid.E.copy())

    def save_cells(self, grid: Union[grids.Grid1D, grids.Grid1D3V, grids.Grid2D]):
        self.x.append(grid.x.copy())

    @staticmethod
    def read(dirname, dataname):
        """
        Read the list in dirname/dataname.pkl
        Returns a python list [] with the results
        """
        file = os.path.join(dirname, f"{dataname}.pkl")
        with open(file, "rb") as f:
            data = pickle.load(f)
        return data

    def write(self, dirname, parameters: Parameters):
        """
        Save the results in a file under "dirname/".
        Each variable gets its own file.
        Currently we store everything in memory until the end, then we write
        -> inefficient when datasize increases too much.
        """
        os.makedirs(dirname, exist_ok=True)

        file = os.path.join(dirname, "parameters.txt")
        with open(file, "w") as f:
            f.write(repr(parameters))

        names_dict = {
            "t": self.t,
            "KE": self.KE,
            "PE": self.PE,
            "TE": self.TE,
            "n_e": self.n_e,
            "n_i": self.n_i,
            "x_e": self.x_e,
            "v_e": self.v_e,
            "x_i": self.x_i,
            "v_i": self.v_i,
            "E": self.E,
            "x": self.x,
        }

        for dataname, data in names_dict.items():
            file = os.path.join(dirname, f"{dataname}.pkl")
            with open(file, "wb") as f:
                pickle.dump(data, f)


class Results1D(Results):
    def __init__(self):
        super().__init__()
        # 1D results store the potential phi(x)
        self.phi = []

    def save_fields(self, grid: grids.Grid1D):
        super().save_fields(grid)
        self.phi.append(grid.phi.copy())

    def write(self, dirname, p: Parameters):
        super().write(dirname, p)

        names_dict = {"phi": self.phi}

        for dataname, data in names_dict.items():
            file = os.path.join(dirname, f"{dataname}.pkl")
            with open(file, "wb") as f:
                pickle.dump(data, f)


class ResultsND(Results):
    def __init__(self):
        super().__init__()
        # ND results store the magnetic field B(x) and current density J(x)
        self.J = []
        self.B = []
        self.rho = []
        self.gauss_l2 = []
        self.gauss_linf = []
        self.continuity_l2 = []
        self.continuity_linf = []
        self.electric_longitudinal_energy = []
        self.electric_transverse_energy = []
        self.magnetic_energy = []

    def save_fields(self, grid: Union[grids.Grid1D3V, grids.Grid2D]):
        super().save_fields(grid)
        self.J.append(grid.J.copy())
        self.B.append(grid.B.copy())
        self.rho.append(grid.rho.copy())
        volume_element = grid.dx if grid.E.ndim == 2 else grid.dx * grid.dx
        self.gauss_l2.append(
            np.sqrt(volume_element * np.sum(grid.gauss_residual**2))
        )
        self.gauss_linf.append(np.max(np.abs(grid.gauss_residual)))
        self.continuity_l2.append(
            np.sqrt(volume_element * np.sum(grid.continuity_residual**2))
        )
        self.continuity_linf.append(np.max(np.abs(grid.continuity_residual)))
        if grid.E.ndim == 2:
            longitudinal = np.sum(grid.E[:, 0] ** 2)
            transverse = np.sum(grid.E[:, 1:] ** 2)
        else:
            longitudinal = np.sum(grid.E**2)
            transverse = 0.0
        self.electric_longitudinal_energy.append(
            0.5 * eps_0 * volume_element * longitudinal
        )
        self.electric_transverse_energy.append(
            0.5 * eps_0 * volume_element * transverse
        )
        self.magnetic_energy.append(
            0.5 / mu_0 * volume_element * np.sum(grid.B**2)
        )

    def write(self, dirname, p: Parameters):
        super().write(dirname, p)

        names_dict = {
            "J": self.J,
            "B": self.B,
            "rho": self.rho,
            "gauss_l2": self.gauss_l2,
            "gauss_linf": self.gauss_linf,
            "continuity_l2": self.continuity_l2,
            "continuity_linf": self.continuity_linf,
            "electric_longitudinal_energy": self.electric_longitudinal_energy,
            "electric_transverse_energy": self.electric_transverse_energy,
            "magnetic_energy": self.magnetic_energy,
        }

        for dataname, data in names_dict.items():
            file = os.path.join(dirname, f"{dataname}.pkl")
            with open(file, "wb") as f:
                pickle.dump(data, f)
