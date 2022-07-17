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
        U0 = 3,
        nbins=100,
        hist_range = None,
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
        self.U0 = self.U0 * self.kT
        self.nbins = nbins
        self.hist_range = hist_range
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

    def compute_distribution(self):
        if self.number_trjs == 1:
            self.histogram, self.sim_edges = np.histogram(self.x, bins=self.nbins)
        else:
            if self.hist_range == None:
                raise ValueError('hist_range required for computing the distribution!')
            else:
                hist_dummy, self.sim_edges = np.histogram(self.x, bins=self.nbins, range=self.hist_range)
                self.histogram += (hist_dummy / len(self.x))

    def compute_free_energy(self):
        self.fe_pos =(self.sim_edges[1:] + self.sim_edges[:-1]) / 2
        self.fe_sim = -self.kT * np.log(self.histogram)
        self.fe_sim -= np.min(self.fe_sim)

    def plot_fe(self):
        plt.plot(self.fe_pos, self.fe_sim / self.kT, label = 'PMF from sim.')
        plt.plot(self.fe_pos, force_PMF(self.fe_pos, self.U0) / self.kT, label = 'PMF from input')
        plt.ylim(ymax = 1.5 * self.U0 / self.kT)
        plt.ylabel('free energy [kT]')
        plt.xlabel('x')
        plt.legend()
        plt.show()

    def save_fe(self):
        array = np.concatenate(
            (self.fe_pos.reshape((self.nbins,1)), 
            self.fe_sim.reshape((self.nbins,1))), 
            axis = 1)
        np.save(self.path_to_save + 'traj_fe', array)

    def NL_GLE_integrate(self):
        self.parse_input()
        self.gen_initial_values()
        print('Integrating using BAOAB scheme')
        for trj in range(self.number_trjs):
            self.x, self.v, self.x_vec, self.v_vec = BAOAB_integrator(
                self.trj_len,
                self.x_vec,
                self.v_vec,
                self.masses, 
                self.coupling_ks,
                self.alphas,
                self.gammas, 
                self.dt)
            self.compute_distribution()
            if self.save:
                np.save(self.path_to_save + 'traj_'+str(trj), self.x)
                np.save(self.path_to_save + 'vel_'+str(trj), self.v)

        self.compute_free_energy()
        if self.plot:
            self.plot_fe()
        if self.save:
                self.save_fe()