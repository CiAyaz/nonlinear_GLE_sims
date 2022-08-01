import numpy as np
from numba import njit
from math import sqrt, exp, atan, pi

@njit()
def PMF(x, U0):
    return U0 * (x ** 2 - 1) ** 2

@njit()
def force_PMF(x, U0):
    return -4 * U0 * (x ** 3 - x)

@njit()
def DarctanDxSq(x, x0):
    return (pi / (1 + (pi * (x - x0)) ** 2))

@njit()
def gamma21(x):
    return (DarctanDxSq(x, 0.5) + DarctanDxSq(x, -0.5) + 1)

@njit()
def sigma11(x, gammas):
    return sqrt(gammas[0,0])

@njit()
def sigma21(x, gammas):
    return (gammas[0,1] + gamma21(x)) / sigma11(x, gammas) / 2

@njit()
def sigma22(x, gammas):
    return sqrt(gammas[1,1] - sigma21(x,gammas) ** 2)

@njit()
def force(x, v_vec, gammas, U0):
    output = np.zeros_like(v_vec)
    output[0] = force_PMF(x, U0)
    output[0] -= gammas[0,0] * v_vec[0]
    output[0] -= gammas[0,1] * v_vec[1]
    output[1] -= gamma21(x) * v_vec[0]
    output[1] -= gammas[1,1] * v_vec[1]

    return output

@njit()
def vel_coupl_leapfrog_integrator(nsteps, x, v_vec, masses, gammas, dt, kT, U0):
    
    nvels = len(v_vec)
    th = 0.5 * dt
    tm = np.zeros_like(v_vec)
    xi_factor = np.zeros_like(v_vec)
    for nvel in range(nvels):
        tm[nvel] = dt / masses[nvel]
        xi_factor[nvel] = sqrt(2 * kT * dt) / masses[nvel] 
    
    trajx=np.zeros(nsteps)
    trajv=np.zeros(nsteps)

    for i in range(nsteps):
        xi = np.random.randn(2)

        trajx[i] = x
        trajv[i] = v_vec[0]
        
        x += th * v_vec[0]

        f = force(x, v_vec, gammas, U0)

        v_vec[0] += (tm[0] * f[0] + xi_factor[0] * sigma11(x, gammas) * xi[0])
        v_vec[1] += (tm[1] * f[1] + xi_factor[1] * (sigma21(x, gammas) * xi[0] + sigma22(x, gammas) * xi[1]))
        
        x += th * v_vec[0]

    return trajx, trajv, x, v_vec
