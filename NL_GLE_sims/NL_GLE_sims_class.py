import numpy as np
from scipy.interpolate import CubicSpline
from sympy import ShapeError
import matplotlib.pyplot as plt
from NL_GLE_sims.tools import *


class NL_GLE_sims():
    """
    Main class for nonlinear GLE simulations.
    """

    def __init__(self, 
        free_energy, 
        alpha_prime_sq,
        gammas, 
        masses,
        coupling_ks,  
        dt = 0.01,
        traj_length = int(1e6), 
        number_trajs = 1,
        kT = 2.494,
        plot=True,
        save=False,
        path_to_save='./'
        ):
        self.dt = dt
        self.trj_len = traj_length
        self.number_trjs = number_trajs
        self.masses = masses
        self.gammas = gammas
        self.coupling_ks = coupling_ks
        if free_energy.shape[1] != 2:
            raise ShapeError('free energy array must contain two columns (positions, energies)!')
        else:
            self.fe = free_energy[:,1]
            self.edges = free_energy[:,0]
        self.kT=kT
        self.plot = plot
        self.save = save
        self.path_to_save = path_to_save

    def parse_input(self):
        for input in [self.gammas, self.masses, self.coupling_ks]:
            if not isinstance(input, (np.ndarray, list)):
                raise TypeError('Give gammas, ks. and masses as list of scalars or numpy.ndarray!')

        if isinstance(self.gammas, list):
            self.gammas = np.array(self.gammas)
        
        if isinstance(self.coupling_ks, list):
            self.coupling_ks = np.array(self.coupling_ks)
        
        if isinstance(self.masses, list):
            self.masses = np.array(self.masses)

        if len(self.masses) != len(self.gammas) + 1 or len(self.gammas) != len(self.coupling_ks):
            raise ShapeError('length masses = length gammas + 1 = length ks + 1 is not fulfilled!')