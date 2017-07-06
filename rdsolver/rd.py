"""
Solves system of differential equations of the form
dc_i/dt = D_i (\partial^2/\partial x^2 + \partial^2/\partial y^2) c_i
             + \beta_i + \gamma_i c_i + f_i(c_1, c_2, ...).

Specifically, the user specifies the diffusion coefficients, the values
of the paramters beta and gamma for each chemical species, and a
function giving the nonlinear terms of the chemical reaction, which
can in general be a function of time.
"""
import numpy as np

from . import utils


def _check_and_update_inputs(D, n, beta, gamma, f_args, L):
    """
    Check inputs and update parameters for convenient use.
    """

    # Make sure beta, gamma, and D are all arrays
    if np.isscalar(D):
        beta = np.array([D])

    if np.isscalar(beta):
        beta = np.array([beta])

    if np.isscalar(gamma):
        beta = np.array([gamma])

    if not (len(D) == len(beta) == len(gamma)):
        raise RuntimeError('D, beta, and gamma must all be the same length.')

    # Perform further checks and updates
    D = _check_D(D)
    n = _check_n_gridpoints(n)
    beta = _update_beta(beta, n)
    gamma = _update_gamma(gamma, n)
    f_args = _check_f_args(f_args)
    L = _check_L(L)

    return D, n, beta, gamma, f_args, L


def _update_beta(beta, n):
    """
    Convert beta to something convenient add to flattened
    concentrations.

    Parameters
    ----------
    beta : nd_array or None
        If not None, array of autoproduction rates for each chemical
        species.
    n : 2-tuple of ints
        n[0] is the number of rows of differencing grid.
        n[1] is the number of columns of differencing grid.

    Returns
    -------
    output : nd_array, shape (len(beta) * n[0] * n[1])
        Array that can be conveniently added to concentrations.
    """
    if beta is None:
        return None

    # Convert scalar to array
    if np.isscalar(beta):
        beta = np.array([beta])

    # Make sure they're floats
    beta = beta.astype(float)

    return np.concatenate([np.array([beta_i] * n[0]*n[1]) for beta_i in beta])


def _update_gamma(gamma, n):
    """
    Convert gamma to something convenient multiply flattened
    concentrations by.

    Parameters
    ----------
    gamme : nd_array or None
        If not None, array of autogrowth rate constants for each
        chemical species.
    n : 2-tuple of ints
        n[0] is the number of rows of differencing grid.
        n[1] is the number of columns of differencing grid.

    Returns
    -------
    output : nd_array, shape (len(gamma) * n[0] * n[1])
        Array that can be conveniently added to concentrations.
    """
    return _update_beta(gamma, n)


def _check_L(L):
    """
    Check physical lengths of domain.
    """
    if L is None:
        return (2*np.pi, 2*np.pi)

    # Length must be 2-tuple
    if type(L) in [list, np.ndarray]:
        L = tuple(L)

    if type(L) != tuple or len(L) != 2:
        raise Runtimerror('`L` must be a 2-tuple.')

    # Make sure the lengths are float
    return (float(L[0]), float(L[1]))


def _check_f_args(f_args):
    """
    Check arguments for reaction function.
    """
    if f_args is None:
        return tuple()

    if type(f_args) in [list, np.ndarray]:
        return tuple(f_args)


def _check_D(n):
    """
    Check diffusion coefficient.
    """

    if np.isscalar(D):
        D = np.array([D])

    # Number of grid points must be tuple of ints
    if type(D) in [list, tuple]:
        D = np.array(D)

    if len(D.shape != 1):
        raise RuntimeError('D must be a one-dimensional array.')

    # Make sure it is a float
    D = D.astype(float)

    return D.astype(float)


def _check_n_gridpoints(n):
    """
    Check number of grid points meet requirements.
    """
    # Number of grid points must be tuple of ints
    if type(n) in [list, np.ndarray]:
        n = tuple(n)

    if type(n) != tuple or len(n) != 2:
        raise Runtimerror('`n` must be a 2-tuple.')

    # Make sure the number of grid points are ints
    if type(n[0]) != int or type(n[1]) != int:
        raise RuntimeError('Number of grid points must be integer.')

    # Make sure they are even
    if n[0] % 2 != 0 or n[1] % 2 != 0:
        raise RuntimeError('Number of grid points must be even.')

    return n


def dc_dt(c, t, D, n, beta=None, gamma=None, f=None, f_args=(), L=None,
          diff_multiplier=None):
    """
    Right hand side of R-D dynamics in real space.
    """
    rhs = np.zeros_like(c)
    n_tot = n[0] * n[1]

    # Compute diffusive terms
    for i, d in enumerate(D):
        i0 = i * n_tot
        i1 = (i+1) * n_tot
        rhs[i0:i1] += d * utils.laplacian_flat_periodic_2d(c[i0:i1], n, L=None,
                                                           diff_multiplier=None)

    # Add reaction terms
    if beta is not None:
        rhs += beta

    if gamma is not None:
        rhs += gamma * c

    if f is not None:
        rhs += f(c.reshape((len(c) // n_tot, n_tot)), t, *f_args).flatten()

    return rhs


# ##################
# ALL FUNCTIONS BELOW NOT YET WORKING
# ##################

def nonlinear_terms(c, rnl, n_species, rnl_args=()):
    """
    Parameters
    ----------
    c : array_like, shape (n_species * nx * ny, )
        Concentrations at a given time point. Organized where
        first len(c)/n_species entries are flattened array of
        concentrations of first chemical species, next
        len(c)/n_species for for second chemical species, and
        so on.
    rnl : function, rnl(c1, c2, ..., *rnl_args)
        Return the nonlinear terms of the dynamics. The first k
        arguments are the concentrations of teh k different species.
        These may be arrays, all with the same shape. The remaining
        arguments are parameters used in the calculation. This function
        returns an n_species-tuple with the nonlinear terms for the
        dynamics of each species.
    rnl_args : tuple, default empty tuple
        Arguments to pass to rnl.

    Returns
    -------
    output : array_like

    """

    nl_terms = rnl(*c.reshape((n_species, len(c)//n_species)), *rnl_args)
    return np.concatenate(nl_terms)


def imex_2d(c0, t, dt, f_hat, f_hat0, c_hat, D, rl, k2):
    """
    Does Adams-Bashforth/Crank-Nicholson integration of ARDA object.
    """

    rkf45_time_points = np.array([arda.time_points[0],
                                  arda.time_points[0] + dt / 2.0,
                                  arda.time_points[0] + dt])
    rkf45_output = utils.rkf45(
        arda.time_deriv, c0, rkf45_time_points,
        args=(arda,), dt=arda.dt / 10.0)

    # Initialize previous steps from solution
    u = (rkf45_output[:,0], rkf45_output[:,-1])

    # Pull out concentrations and compute FFTs
    c = ([None for i in xrange(arda.n_species)],
         [None for i in xrange(arda.n_species)])
    c_hat = ([None for i in xrange(arda.n_species)],
             [None for i in xrange(arda.n_species)])
    if arda.n_species > 0:
        if arda.nematic:
            start_ind = 2
        else:
            start_ind = 0
        for i in xrange(arda.n_species):
            i0 = arda.na_x * arda.na_y * (i + start_ind)
            i1 = i0 + arda.na_x * arda.na_y
            c[0][i] = \
                arda.initial_condition[i0:i1].reshape((arda.na_y, arda.na_x))
            c[1][i] = u[1][i0:i1].reshape((arda.na_y, arda.na_x))
            c_hat[0][i] = fft2(c[0][i])
            c_hat[1][i] = fft2(c[1][i])


    # Compute initial f_hat
    if arda.nematic:
        f_hat = (arda.nonlin_fun(Q_tilde[0], q[0], Q_tilde_hat[0], q_hat[0],
                                 c[0], arda.time_points[0], arda),
                 arda.nonlin_fun(Q_tilde[1], q[1], Q_tilde_hat[1], q_hat[1],
                                 c[1], arda.time_points[0] + arda.dt, arda))
    else:
        f_hat = (arda.nonlin_fun(c[0], arda.time_points[0], arda),
                 arda.nonlin_fun(c[1], arda.time_points[0] + arda.dt, arda))

    # Set up return variables
    u_sol = [arda.initial_condition]
    t = arda.time_points[0] + arda.dt
    t_sol = [arda.time_points[0]]
    i_max = len(arda.time_points)
    i = 1

    # Take the time steps
    while i < i_max:
        while t < arda.time_points[i]:
            c_hat_step = \
                stepper(arda, arda.dt, 1.0, f_hat, c_hat)

            # Update hat variable
            c_hat = (c_hat[1], c_hat_step)

            # Update real space variables
            c = (c[1], [None] * arda.n_species)
            for j in xrange(arda.n_species):
                c[1][j] = ifft2(c_hat_step[j]).real

            # Increment
            t += arda.dt
            f_hat = (f_hat[1], arda.nonlin_fun(c[1], t, arda))

        t_sol.append(t)

        # Pull out current solution and flatten for output
        if arda.nematic:
            u = np.empty((2 + arda.n_species) * arda.na_x * arda.na_y)
            u[:arda.na_x*arda.na_y] = Q_tilde[1].flatten()
            u[arda.na_x*arda.na_y:2*arda.na_x*arda.na_y] = q[1].flatten()
            for j in xrange(arda.n_species):
                i0 = arda.na_x * arda.na_y * (j + 2)
                i1 = i0 + arda.na_x * arda.na_y
                u[i0:i1] = c[1][j].flatten()
        else: # Isotropic
            u = np.empty(arda.n_species * arda.na_x * arda.na_y)
            for j in xrange(arda.n_species):
                i0 = arda.na_x * arda.na_y * j
                i1 = i0 + arda.na_x * arda.na_y
                u[i0:i1] = c[1][j].flatten()

        if np.isnan(u).any() != 0:
            raise ValueError('Solution blew up! Try reducing dt.')

        u_sol.append(u)
        i += 1

    # Interpolate solution
    return ode.ode_int.interpolate_solution(
        np.array(u_sol).transpose(), np.array(t_sol), arda.time_points)


def cnab2_step(dt_current, dt0, f_hat, f_hat0, c_hat, D, rl, k2):
    """
    Takes a Crank-Nicolson/Adams-Bashforth (2nd order) step for
    RD system in Fourier space.

    Parameters
    ----------
    dt_current : float
        Current time step
    dt0 : float
        Previous time step
    f_hat : array_like, shape (n_species * total_n_grid_points, )
        Current FFT of nonlinear part of dynamics. Organized where
        first len(c_hat)/len(D) entries are flattened array of
        concentrations of first chemical species, next
        len(c_hat)/len(D) for for second chemical species, and
        so on.
    f_hat0 : array_like, shape (n_species * total_n_grid_points, )
        FFT of nonlinear part of dynamics from previous time step.
    c_hat : array_like, shape (n_species * total_n_grid_points, )
        Current FFT of concentrations.
    D : array_like, shape (n_species, )
        Array of diffusion coefficients for all chemical species.
    rl : array_like, shape (n_species, n_species)
        rl[i,j] is the rate constant to the linear term in reaction
        dynamics describing species i for chemical species j. As an
        example, if all species have simple decay term, then rl is
        diagonal.
    k2 : array_like, shape (total_n_grid_points, )
        The square of the wave number for the grid points as a
        flattened array

    Returns
    -------
    output : array_like, shape (n_species * total_n_grid_points, )
        Updated FFT of concentrations.
    """

    omega = dt_current / dt0
    n = len(c_hat) // len(D)

    new_c = np.copy(c)

    for i, d in enumerate(D):
        i0 = i*n
        i1 = (i+1)*n
        f = f_hat[i0:i1]
        f0 = f_hat0[i0:i1]
        c = c_hat[i0:i1]

        # CHECK THIS TO MAKE SURE LINEAR TERM IS DONE CORRECTLY
        new_c[i0:i1] = c + dt_current * ((1 + omega/2) * f - omega/2 * f
                + (d * k2 - rl.sum(axis=1)) * c)

    return new_c
