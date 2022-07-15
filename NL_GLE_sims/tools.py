import numpy as np
from numba import njit
from math import sqrt, exp

@njit()
def force_PMF(x, U0):
    return -4 * U0 * (x ** 3 - x)

@njit()
def alpha1(x, alpha0):
    return 0

@njit()
def force_coupling1(x_vector):
    return 0

@njit()
def force(x_vector):
    return 0


@njit()
def BAOAB_integrator(force, nsteps, x_init, v_init, masses, dt=0.001, friction=0.1,
kT=2.494):
    """Langevin integrator for initial value problems
    This function implements the BAOAB algorithm of Benedict Leimkuhler
    and Charles Matthews. See J. Chem. Phys. 138, 174102 (2013) for
    further details.
    Arguments:
        force (function): computes the forces of a single configuration
        nsteps (int): number of integration steps
        x_init (numpy.ndarray(n, d)): initial configuration
        v_init (numpy.ndarray(n, d)): initial velocities
        masses (numpy.ndarray(n)): particle masses
        dt (float): time step for the integration
        friction (float): damping term, use zero if not coupled
        kT (float): thermal energy
    Returns:
        x (numpy.ndarray(nsteps + 1, n, d)): configuraiton trajectory
        v (numpy.ndarray(nsteps + 1, n, d)): velocity trajectory
    """
    nvars = len(x_init)
    th = 0.5 * dt
    thm = np.zeros_like(x_init)
    edt = np.zeros_like(x_init)
    sqf = np.zeros_like(x_init)
    for nvar in range(nvars):
        thm[nvar] = 0.5 * dt / masses[nvar]
        edt[nvar] = exp(-friction[nvar] * dt)
        sqf[nvar] = sqrt((1.0 - edt[nvar] ** 2) / (masses[nvar] / kT))

    x = np.zeros(nsteps)
    v = np.zeros(nsteps)
    f = force(x_init)

    for i in range(nsteps):

        x[i] = x_init[0]
        v[i] = v_init[0]

        for nvar in range(nvars):
            v_init[nvar] += thm[nvar] * f[nvar]
            x_init[nvar] += th[nvar] * v_init[nvar]
            v_init[nvar] *= edt[nvar]
            v_init[nvar] += sqf[nvar] * np.random.randn()
            x_init[nvar] += th[nvar] * v_init[nvar]
        f = force(x_init)
        for nvar in range(nvars):
            v_init[nvar] += thm[nvar] * f[nvar]

    return x, v