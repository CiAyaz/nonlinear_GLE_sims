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
def force_couplingx_linear(x, y, k0):
    return -k0 * (x - y)

@njit()
def force_couplingy_linear(x, y, k0):
    return -k0 * (y - x)

@njit()
def force_linear(x_vec, couplings, U0, nvars):
    output = np.zeros(nvars)
    output[0] = force_PMF(x_vec[0],U0)
    output[0] += force_couplingx_linear(x_vec[0], x_vec[2], couplings[1])
    output[0] += force_couplingx_linear(x_vec[0], x_vec[1], couplings[0])
    output[1] = force_couplingy_linear(x_vec[0], x_vec[1], couplings[0])
    output[2] = force_couplingy_linear(x_vec[0], x_vec[2], couplings[1])

    return output



@njit()
def BAOAB_integrator(nsteps, x_vec, v_vec, masses, couplings, alphas, friction, dt, kT, U0):
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
            xi = np.random.normal(0., 1.)
            v_vec[nvar] += thm[nvar] * f[nvar]
            x_vec[nvar] += th * v_vec[nvar]
            v_vec[nvar] *= edt[nvar]
            v_vec[nvar] += sqf[nvar] * xi
            x_vec[nvar] += th * v_vec[nvar]
        f = force(x_vec, couplings, alphas, U0, nvars)
        for nvar in range(nvars):
            v_vec[nvar] += thm[nvar] * f[nvar]

    return x, v, x_vec, v_vec

@njit()
def leapfrog_integrator(nsteps, x_vec, v_vec, masses, couplings, alphas, friction, dt, kT, U0):
    
    nvars = len(x_vec)
    th = 0.5 * dt
    tm = np.zeros_like(x_vec)
    tgm = np.zeros_like(x_vec)
    xi_factor = np.zeros_like(x_vec)
    for nvar in range(nvars):
        tm[nvar] = dt / masses[nvar]
        tgm[nvar] = - dt * friction[nvar] / masses[nvar]
        xi_factor[nvar] = sqrt(2 * kT * friction[nvar] * dt) / masses[nvar]
    
    x=np.zeros(nsteps)
    v=np.zeros(nsteps)

    for i in range(nsteps):

        x[i] = x_vec[0]
        v[i] = v_vec[0]
        
        for nvar in range(nvars):
            x_vec[nvar] += v_vec[nvar] * th
        f = force(x_vec, couplings, alphas, U0, nvars)
        for nvar in range(nvars):
            xi = np.random.randn()
            v_vec[nvar] += tgm[nvar] * v_vec[nvar]  
            v_vec[nvar] += (tm[nvar] * f[nvar] + xi_factor[nvar] * xi)
            x_vec[nvar] += th * v_vec[nvar]

    return x, v, x_vec, v_vec


@njit()
def leapfrog_Euler_integrator(
    nsteps, x_vec, v_vec, masses, couplings, alphas, friction, dt, kT, U0):
    
    nvars = len(x_vec)
    th = 0.5 * dt
    tm = dt / masses[0]
    tg = np.zeros_like(x_vec)
    xi_factor = np.zeros_like(x_vec)
    for nvar in range(1,nvars):
        tg[nvar] = dt / friction[nvar]
        xi_factor[nvar] = sqrt(2 * kT * dt / friction[nvar])
    
    x=np.zeros(nsteps)
    v=np.zeros(nsteps)

    for i in range(nsteps):

        x[i] = x_vec[0]
        v[i] = v_vec[0]
        
        f0 = force(x_vec, couplings, alphas, U0, nvars) 
        x_vec[0] += v_vec[0] * th
        fh = force(x_vec, couplings, alphas, U0, nvars)
        v_vec[0] += tm * fh[0]
        x_vec[0] += th * v_vec[0]
        for nvar in range(1,nvars):
            xi = np.random.randn()  
            x_vec[nvar] += (tg[nvar] * f0[nvar] + xi_factor[nvar] * xi)

    return x, v, x_vec, v_vec


@njit()
def Runge_Kutta_integrator(
    nsteps, x_vec, v_vec, masses, couplings, alphas, friction, dt, kT, U0):
    """
    Integrator for a Markovian Embedding with exponentially
    decaying memory kernels, characterized by friction gamma[i]
    and memory time tgammas[i]. Uses spline rep from pot_edges
    and amatrix. Restarts sim from pos x0 and velocitiy v
    and position of the overdamped orth. dof at R.
    """

    # relevant constants
    nvars = len(x_vec)
    xi_factor = np.zeros(nvars)
    xi = np.zeros(nvars)
    xi_factor[0] = sqrt(2 * kT * friction[0] * dt)
    for y in range(1, nvars):
        xi_factor[y] = sqrt(2 * kT / friction[y] * dt)

    mass = masses[0] 
    
    # runge kutta step factors
    RK = np.array([0.5, 0.5, 1.])

    # arrays to store temp data
    vars = np.zeros((4, nvars))
    vs = np.zeros(4)
    vars[0] = x_vec
    vs[0] = v_vec[0]

    k = np.zeros((4, nvars))
    kv = np.zeros(4)
    
    # trajectory array
    x = np.zeros(nsteps)
    v = np.zeros(nsteps)

    for step in range(nsteps):
        # draw random force
        xi[1:] = np.random.normal(0., 1., nvars - 1)
        # first 3 runge kutta steps
        for rk in range(3):
            ft = force(vars[rk], couplings, alphas, U0, nvars)
            k[rk, 0] = dt * vs[rk]
            kv[rk] = (dt * (ft[0]
            - friction[0] * vs[rk]) + xi_factor[0] * xi[0]) / mass
            # orhtogonal degrees of freedom
            for y in range(1, nvars):
                k[rk, y] = (dt * ft[y] / friction[y]
                + xi_factor[y]*xi[y])
                vars[rk + 1, y] = vars[0, y] + RK[rk] * k[rk, y]
            # variable of interest
            vars[rk + 1, 0] = vars[0, 0] + RK[rk] * k[rk, 0]
            vs[rk + 1] = vs[0] + RK[rk] * kv[rk]

        # last runge kutta step
        ft = force(vars[3], couplings, alphas, U0, nvars)
        k[3, 0] = dt * vs[3]
        k[3, 1] = (dt * (ft[0]
        - friction[0] * vs[3]) + xi_factor[0] * xi[0]) / mass
        # orhtogonal degrees of freedom
        for y in range(1, nvars):
            k[3, y] = (dt * ft[y] / friction[y]
            + xi_factor[y]*xi[y])
            vars[0, y] += (k[0, y] + 2 * k[1, y] + 2 * k[2, y] + k[3, y]) / 6
        # variable of interest
        vars[0, 0] += (k[0, 0] + 2 * k[1, 0] + 2 * k[2, 0] + k[3, 0]) / 6
        vs[0] += (kv[0] + 2 * kv[1] + 2 * kv[2] + kv[3]) / 6

        x[step] = vars[0, 0]
        v[step] = vs[0]

    x_vec = vars[0]
    v_vec[0] = vs[0]

    return x, v, x_vec, v_vec



@njit()
def leapfrog_integrator_linear_coupling(nsteps, x_vec, v_vec, masses, couplings, friction, dt, kT, U0):
    
    nvars = len(x_vec)
    th = 0.5 * dt
    tm = np.zeros_like(x_vec)
    tgm = np.zeros_like(x_vec)
    xi_factor = np.zeros_like(x_vec)
    for nvar in range(nvars):
        tm[nvar] = dt / masses[nvar]
        tgm[nvar] = - dt * friction[nvar] / masses[nvar]
        xi_factor[nvar] = sqrt(2 * kT * friction[nvar] * dt) / masses[nvar]
    
    x=np.zeros(nsteps)
    v=np.zeros(nsteps)

    for i in range(nsteps):

        x[i] = x_vec[0]
        v[i] = v_vec[0]
        
        for nvar in range(nvars):
            x_vec[nvar] += v_vec[nvar] * th
        f = force_linear(x_vec, couplings, U0, nvars)
        for nvar in range(nvars):
            xi = np.random.randn()
            v_vec[nvar] += tgm[nvar] * v_vec[nvar]  
            v_vec[nvar] += (tm[nvar] * f[nvar] + xi_factor[nvar] * xi)
            x_vec[nvar] += th * v_vec[nvar]

    return x, v, x_vec, v_vec