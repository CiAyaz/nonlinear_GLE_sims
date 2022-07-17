import numpy as np
from numba import njit
from math import sqrt, exp, atan, pi

@njit()
def force_PMF(x, U0):
    return -4 * U0 * (x ** 3 - x)

@njit()
def alpha1(x, alpha0):
    return alpha0 * atan(pi * (x - 0.5))

@njit()
def alpha2(x, alpha0):
    return alpha0 * atan(pi * (x + 0.5))

@njit()
def dalpha1_dx(x, alpha0):
    return alpha0 * pi / (1 + (pi * (x - 0.5)) ** 2)

@njit()
def dalpha2_dx(x, alpha0):
    return alpha0 * pi / (1 + (pi * (x + 0.5)) ** 2)

@njit()
def force_couplingx(func, dfunc_dx, x, y, k0, alpha0):
    return -k0 * dfunc_dx(x,alpha0) * (func(x,alpha0) - y)

@njit()
def force_couplingy(func, x, y, k0, alpha0):
    return -k0 * (y - func(x,alpha0))

@njit()
def force(x_vec, couplings, alphas, U0, nvars):
    output = np.zeros(nvars)
    output[0] = force_PMF(x_vec[0],U0)
    output[0] += force_couplingx(alpha1, dalpha1_dx, x_vec[0], x_vec[1], couplings[0], alphas[0])
    output[0] += force_couplingx(alpha2, dalpha2_dx, x_vec[0], x_vec[2], couplings[1], alphas[1])
    output[1] = force_couplingy(alpha1, x_vec[0], x_vec[1], couplings[0], alphas[0])
    output[2] = force_couplingy(alpha2, x_vec[0], x_vec[2], couplings[1], alphas[1])

    return output


@njit()
def BAOAB_integrator(nsteps, x_vec, v_vec, masses, couplings, alphas, friction, dt, kT=2.494, U0=3):
    """Langevin integrator for initial value problems
    This function implements the BAOAB algorithm of Benedict Leimkuhler
    and Charles Matthews. See J. Chem. Phys. 138, 174102 (2013) for
    further details.
    Arguments:
        force (function): computes the forces of a single configuration
        nsteps (int): number of integration steps
        x_vec (numpy.ndarray(n, d)): initial configuration
        v_vec (numpy.ndarray(n, d)): initial velocities
        masses (numpy.ndarray(n)): particle masses
        dt (float): time step for the integration
        friction (float): damping term, use zero if not coupled
        kT (float): thermal energy
    Returns:
        x (numpy.ndarray(nsteps + 1, n, d)): configuraiton trajectory
        v (numpy.ndarray(nsteps + 1, n, d)): velocity trajectory
    """
    U0 *= kT
    nvars = len(x_vec)
    th = 0.5 * dt
    thm = np.zeros_like(x_vec)
    edt = np.zeros_like(x_vec)
    sqf = np.zeros_like(x_vec)
    for nvar in range(nvars):
        thm[nvar] = 0.5 * dt / masses[nvar]
        edt[nvar] = exp(-friction[nvar] * dt)
        sqf[nvar] = sqrt((1.0 - edt[nvar] ** 2) / (masses[nvar] / kT))

    x = np.zeros(nsteps)
    v = np.zeros(nsteps)
    f = force(x_vec, couplings, alphas, U0, nvars)

    for i in range(nsteps):

        x[i] = x_vec[0]
        v[i] = v_vec[0]

        for nvar in range(nvars):
            v_vec[nvar] += thm[nvar] * f[nvar]
            x_vec[nvar] += th[nvar] * v_vec[nvar]
            v_vec[nvar] *= edt[nvar]
            v_vec[nvar] += sqf[nvar] * np.random.randn()
            x_vec[nvar] += th[nvar] * v_vec[nvar]
        f = force(x_vec, couplings, alphas, U0, nvars)
        for nvar in range(nvars):
            v_vec[nvar] += thm[nvar] * f[nvar]

    return x, v, x_vec, v_vec