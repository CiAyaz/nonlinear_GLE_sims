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
        alphas = None, 
        dt = 0.01,
        traj_length = int(1e6), 
        number_trajs = 1,
        kT = 2.494,
        U0 = 3,
        nbins=100,
        hist_range = None,
        plot = True,
        save = False,
        path_to_save = './',
        integrator = 'leap_frog',
        non_local = False
        ):
        self.dt = dt
        self.trj_len = traj_length
        self.number_trjs = number_trajs
        self.masses = masses
        self.gammas = gammas
        self.coupling_ks = coupling_ks
        self.alphas = alphas
        self.kT=kT
        self.U0 = U0 * self.kT
        self.nbins = nbins
        self.hist_range = hist_range
        self.plot = plot
        self.save = save
        self.path_to_save = path_to_save
        self.integrator = integrator
        self.non_local = non_local
        self.x_vec = np.array([])
        self.v_vec = np.array([])

    def parse_input(self):
        if not self.non_local:
            for input in [self.gammas, self.masses, self.coupling_ks, self.alphas]:
                if not isinstance(input, (np.ndarray, list)):
                    raise TypeError('Give gammas, ks, alphas and masses as list of scalars or numpy.ndarray!')
            
            if isinstance(self.alphas, list):
                self.alphas = np.array(self.alphas)

            if len(self.alphas) != len(self.coupling_ks):
                raise ShapeError('length alphas = length coupling_ks is not fulfilled!')
        else:
            for input in [self.gammas, self.masses, self.coupling_ks]:
                if not isinstance(input, (np.ndarray, list)):
                    raise TypeError('Give gammas, ks and masses as list of scalars or numpy.ndarray!')


        if isinstance(self.gammas, list):
            self.gammas = np.array(self.gammas)
        
        if isinstance(self.coupling_ks, list):
            self.coupling_ks = np.array(self.coupling_ks)
        
        if isinstance(self.masses, list):
            self.masses = np.array(self.masses)

        if len(self.masses) != len(self.gammas):
            raise ShapeError('length masses = length gammas is not fulfilled!')
        if len(self.gammas) != len(self.coupling_ks) + 1:
            raise ShapeError('length gammas = length coupling_ks + 1 is not fulfilled!')

    def gen_initial_values(self): 
        self.x_vec = np.zeros_like(self.masses)
        self.v_vec = np.zeros_like(self.masses)
        self.x_vec[0] = -1.
        for index, coupling in enumerate(self.coupling_ks):
            self.x_vec[1+index] = np.random.normal(self.x_vec[0], np.sqrt(self.kT / coupling))
        if not self.integrator == "RK":
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
        plt.plot(self.fe_pos, PMF(self.fe_pos, self.U0) / self.kT, label = 'PMF from input')
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

    def print_vals(self):
        if not self.integrator == "RK":
            print('epsilon = %.3g,   %.3g'%(np.sqrt(self.masses[1] / self.masses[0]), 
                                            np.sqrt(self.masses[2] / self.masses[0])))
            self.mem_time = 2 * self.masses[1:] / self.gammas[1:]
            nu_sq = 2 * self.mem_time * self.coupling_ks / self.gammas[1:] - 1
            if any(nu_sq) < 0.:
                print('nu_sq = ', nu_sq)
                raise ValueError('nu squared is negative!')
            self.freq = np.sqrt(nu_sq) / (2 * np.pi * self.mem_time)
            print('memory times = %.3g,   %.3g'%(self.mem_time[0], self.mem_time[1]))
            print('oscillation freq. = %.3g,   %.3g'%(self.freq[0], self.freq[1]))
        else:
            self.mem_time = self.gammas[1:] / self.coupling_ks
            epsilon = self.gammas[1:] / np.sqrt(self.masses[0] * self.coupling_ks)
            print(f"epsilon = {epsilon[0]:.3f},   {epsilon[1]:.3f}")
            print('memory times = %.3g,   %.3g'%(self.mem_time[0], self.mem_time[1]))

    def write_info_file(self):
        if not self.integrator == "RK":
            with open(self.path_to_save+'info.txt', 'a') as f:
                    f.write('epsilon = %.3g, %.3g'%(np.sqrt(self.masses[1] / self.masses[0]), 
                                                    np.sqrt(self.masses[2] / self.masses[0])))
                    f.write('\n')
                    f.write('memory times = %.3g, %.3g'%(self.mem_time[0], self.mem_time[1]))
                    f.write('\n')
                    f.write('oscillation freq. = %.3g, %.3g'%(self.freq[0], self.freq[1]))
        else:
            with open(self.path_to_save+'info.txt', 'a') as f:
                f.write(f'memory times = {self.mem_time[0]:.3f},    {self.mem_time[1]:.3f}')

    def NL_GLE_integrate(self):
        self.parse_input()
        if len(self.x_vec) == 0:
            self.gen_initial_values()
        self.print_vals()
        if not self.non_local:
            if self.integrator == 'BAOAB':
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
                        self.dt,
                        self.kT,
                        self.U0)
                    self.compute_distribution()
                    if self.save:
                        np.save(self.path_to_save + 'traj_'+str(trj), self.x)
                        np.save(self.path_to_save + 'vel_'+str(trj), self.v)

            elif self.integrator == 'RK':
                print('Overdamped coupling using Runge-Kutta integrator')
                for trj in range(self.number_trjs):
                    self.x, self.v, self.x_vec, self.v_vec = Runge_Kutta_integrator(
                        self.trj_len,
                        self.x_vec,
                        self.v_vec,
                        self.masses, 
                        self.coupling_ks,
                        self.alphas,
                        self.gammas, 
                        self.dt,
                        self.kT,
                        self.U0)
                    self.compute_distribution()
                    if self.save:
                        np.save(self.path_to_save + 'traj_'+str(trj), self.x)
                        np.save(self.path_to_save + 'vel_'+str(trj), self.v)
            else:
                print('Integrating using leap-frog scheme')
                for trj in range(self.number_trjs):
                    self.x, self.v, self.x_vec, self.v_vec = leapfrog_integrator(
                        self.trj_len,
                        self.x_vec,
                        self.v_vec,
                        self.masses, 
                        self.coupling_ks,
                        self.alphas,
                        self.gammas, 
                        self.dt,
                        self.kT,
                        self.U0)
                    self.compute_distribution()
                    if self.save:
                        np.save(self.path_to_save + 'traj_'+str(trj), self.x)
                        np.save(self.path_to_save + 'vel_'+str(trj), self.v)
        else:
            print('Integrating using leap-frog scheme')
            for trj in range(self.number_trjs):
                self.x, self.v, self.x_vec, self.v_vec = leapfrog_integrator_linear_coupling(
                    self.trj_len,
                    self.x_vec,
                    self.v_vec,
                    self.masses, 
                    self.coupling_ks,
                    self.gammas, 
                    self.dt,
                    self.kT,
                    self.U0)
                self.compute_distribution()
                if self.save:
                    np.save(self.path_to_save + 'traj_'+str(trj), self.x)
                    np.save(self.path_to_save + 'vel_'+str(trj), self.v)

        self.compute_free_energy()
        if self.plot:
            self.plot_fe()
        if self.save:
                self.save_fe()
                self.write_info_file()

    
    def memory_linear_friction(self, x, t):
        if not self.integrator == "RK":
            if not self.non_local:
                funcs = [dalpha1_dx, dalpha2_dx]
                taus = 2 * self.masses[1:] / self.gammas[1:]
                nus = np.sqrt(2 * self.coupling_ks * taus / self.gammas[1:] - 1)
                ft =  (self.coupling_ks * np.exp(-t / taus) * (np.cos(nus * t / taus)
                        + np.sin(nus * t / taus) / nus) / self.masses[0])
                memory = 0.
                for index, func in enumerate(funcs):
                    memory += func(x, self.alphas[index]) ** 2 * ft[index]
            else:
                taus = 2 * self.masses[1:] / self.gammas[1:]
                nus = np.sqrt(2 * self.coupling_ks * taus / self.gammas[1:] - 1)
                ft =  (self.coupling_ks * np.exp(-t / taus) * (np.cos(nus * t / taus)
                        + np.sin(nus * t / taus) / nus) / self.masses[0])
                memory = np.sum(ft)
                
        else:
            funcs = [dalpha1_dx, dalpha2_dx]
            taus = self.gammas[1:] / self.coupling_ks
            ft =  (self.coupling_ks * np.exp(-t / taus) / self.masses[0])
            memory = 0.
            for index, func in enumerate(funcs):
                memory += func(x, self.alphas[index]) ** 2 * ft[index]
        
        return memory
        
    def hybrid_GammaL(self, t):
        funcs = [dalpha1_dx, dalpha2_dx]
        taus = self.gammas[1:] / self.coupling_ks
        ft =  (self.coupling_ks * np.exp(-t / taus) / self.masses[0])
        pdf_pos = np.linspace(-3,3,1000)
        pdf = np.exp(-PMF(pdf_pos, self.U0))
        averages = []
        for index,func in enumerate(funcs):
            averages.append(np.trapz(pdf * func(pdf_pos, self.alphas[index]) ** 2, pdf_pos))
        memory = 0.
        for index, func in enumerate(funcs):
            memory += averages[index]  * ft[index]
        
        return memory

    def hybrid_D(self, x, t):
        funcs = [dalpha1_dx, dalpha2_dx]
        taus = self.gammas[1:] / self.coupling_ks
        ft =  (self.kT * self.gammas * np.exp(-t / taus) / self.masses[0] ** 2)
        pdf_pos = np.linspace(-3,3,1000)
        pdf = np.exp(-PMF(pdf_pos, self.U0))
        pdf_norm = np.trapz(pdf, pdf_pos)
        pdf /= pdf_norm
        averages = []
        for index,func in enumerate(funcs):
            averages.append(np.trapz(pdf * func(pdf_pos, self.alphas[index]) ** 2, pdf_pos))
        memory = 0.
        for index, func in enumerate(funcs):
            memory += (func(x, self.alphas[index]) ** 2 - averages[index])  * ft[index]
        
        return memory
    

    def memory_function(self, x, t, gle = "hybridGLE", kernel = "GammaL"):
        if gle == "hybridGLE":
            if kernel == "GammaL":
                memory_vec = np.vectorize(self.hybrid_GammaL)
            else:
                memory_vec = np.vectorize(self.hybrid_D)
        else:
            memory_vec = np.vectorize(self.memory_linear_friction)
        return(memory_vec(x,t))