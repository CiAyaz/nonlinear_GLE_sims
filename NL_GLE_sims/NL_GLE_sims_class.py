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
        gammas, 
        masses,
        coupling_ks,
        alphas, 
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
        self.alphas = alphas
        self.kT=kT
        self.plot = plot
        self.save = save
        self.path_to_save = path_to_save

    def parse_input(self):
        for input in [self.gammas, self.masses, self.coupling_ks, self.alphas]:
            if not isinstance(input, (np.ndarray, list)):
                raise TypeError('Give gammas, ks, alphas and masses as list of scalars or numpy.ndarray!')

        if isinstance(self.gammas, list):
            self.gammas = np.array(self.gammas)
        
        if isinstance(self.coupling_ks, list):
            self.coupling_ks = np.array(self.coupling_ks)
        
        if isinstance(self.masses, list):
            self.masses = np.array(self.masses)

        if isinstance(self.alphas, list):
            self.alphas = np.array(self.alphas)

        if len(self.masses) != len(self.gammas):
            raise ShapeError('length masses = length gammas is not fulfilled!')
        if len(self.gammas) != len(self.coupling_ks) + 1:
            raise ShapeError('length gammas = length coupling_ks + 1 is not fulfilled!')
        if len(self.alphas) != len(self.coupling_ks):
            raise ShapeError('length alphas = length coupling_ks is not fulfilled!')

    def gen_initial_values(self): 
        self.x_vec = np.zeros_like(self.masses)
        self.v_vec = np.zeros_like(self.masses)
        self.x_vec[0] = -1.
        for index, coupling in enumerate(self.coupling_ks):
            self.x_vec[1+index] = np.random.normal(self.x_vec[0], np.sqrt(self.kT / coupling))
        for index,m in enumerate(self.masses):
            self.v_vec[index] = np.random.normal(0., np.sqrt(self.kT / m))
