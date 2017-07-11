"""
Solves system of differential equations of the form
dc_i/dt = D_i (\partial^2/\partial x^2 + \partial^2/\partial y^2) c_i
             + \beta_i + \gamma_i c_i + f_i(c_1, c_2, ...).

Specifically, the user specifies the diffusion coefficients, the values
of the paramters beta and gamma for each chemical species, and a
function giving the nonlinear terms of the chemical reaction, which
can in general be a function of time.
"""
import inspect

import numpy as np
import scipy.special

from . import utils

def asdm(d=0.05, mu=1.4):
    """
    Give f, f_args, beta, gamma, and D for the dimensionelss
    activator-substrate depletion model (ASDM).
    """
    D = np.array([d, 1.0])
    beta = np.array([0.0, mu])
    gamma = np.array([-1.0, 0.0])
    def f(c, t, mu):
        a2s = c[0,:,:]**2 * c[1,:,:]
        return np.stack((a2s, -mu * a2s))
    f_args = (mu,)

    return D, beta, gamma, f, f_args

def initial_condition(uniform_conc=None, n=None, L=None, n_bumps=20,
                      bump_width_range=(0.025, 0.1), max_amplitude=0.005):
    """
    Generate initial condition as a small perturbation from uniform c0.

    Parameters
    ----------
    uniform_conc : array_like
        uniform_conc[i] = the uniform concentration for species i about
        with the perturbed initial condition is made.
    n : 2-tuple of ints
        n[0] is the number of rows of differencing grid.
        n[1] is the number of columns of differencing grid.
    L : 2-tuple of float, default (2*pi, 2*pi)
        The physical length of the domain in x and y.
    n_bumps : int, default 20
        Number of bumps to have in the perturbed concentration fields.
    bump_width_range : 2-tuple of floats, default (0.025, 0.1)
        Range of widths of the bumps as a fraction of the width of the
        domain.
    max_amplitude : float, default 0.005
        Maximum amplitude of the perturbation. This is in absolute units
        to allow perturbations from zero.

    Returns
    -------
    output : ndarray, shape (len(uniform_conc), n[0], n[1])
        Initial condition for the concentrations.

    Notes
    -----
    .. Note that all bumps are positive perturbations. This is to
       prevent the possibility of negative concentrations.
    """
    if uniform_conc is None:
        raise RuntimeError('uniform_conc must be given.')

    if n is None:
        raise RuntimeError('n must be given as a 2-tuple.')

    # Get dimensions
    L = _check_L(L)

    # Infer number of species from concnetrations
    if np.isscalar(uniform_conc):
        uniform_conc = [uniform_conc]
    uniform_conc = np.array(uniform_conc)
    n_species = len(uniform_conc)

    # Get grid points
    _, _, _, _, x_grid, y_grid = utils.grid_points_2d(n, L=L)

    # Multiplier factor to smooth out edges
    mult_factor = \
        (1.0 + scipy.special.erf(40.0 * (x_grid - 0.1 * L[0]) / L[0])) / 2.0 \
      * (1.0 - scipy.special.erf(40.0 * (x_grid - 0.9 * L[0]) / L[0])) / 2.0 \
      * (1.0 + scipy.special.erf(40.0 * (y_grid - 0.1 * L[1]) / L[1])) / 2.0 \
      * (1.0 - scipy.special.erf(40.0 * (y_grid - 0.9 * L[1]) / L[1])) / 2.0

    # Make bumps
    c0 = np.ones((n_species, *n))
    for i in range(n_species):
        for j in range(n_bumps):
            x_pos = np.random.rand() * L[0]
            y_pos = np.random.rand() * L[1]
            width = max(L[0], L[1]) * (bump_width_range[0] \
              + (bump_width_range[1] - bump_width_range[0]) * np.random.rand())
            c0[i,:,:] += mult_factor * max_amplitude * np.random.rand() \
                * np.exp(-(x_grid - x_pos)**2 / 2.0 / width**2) \
                * np.exp(-(y_grid - y_pos)**2 / 2.0 / width**2)

    return c0


def dc_dt(c, t, n_species, n, D=None, beta=None, gamma=None, f=None, f_args=(),
          L=None, diff_multiplier=None):
    """
    Right hand side of R-D dynamics in real space.

    Parameters
    ----------
    c : ndarray, shape(n_species * n[0] * n[1], )
        Flattened array of concentrations
    t : float
        Time
    n : 2-tuple of ints
        n[0] is the number of rows of differencing grid.
        n[1] is the number of columns of differencing grid.
    D : array_like, shape (n_species, )
        Array of diffusion coefficients for species.
    beta : array_like, shape (n_species, )
        Array of autoproduction constants.
    gamma : array_like, shape (n_species, )
        Array of degradation constants. Note that negative gamma
        means degradation.
    f : function
        Function to compute the nonlinear terms of the dynamics.
        Call signature f(c, t, *f_args), where c[0] is concentration
        of species 0, c[1] is concentration of species 2, etc.
    f_args : tuple, default ()
    """
    rhs = np.empty_like(c)
    n_tot = n[0] * n[1]

    if D is None:
        D = np.zeros(n_species)

    if beta is None:
        beta = np.zeros(n_species)

    if gamma is None:
        gamma = np.zeros(n_species)

    # Linear terms of right-hand side
    for i, dbg in enumerate(zip(D, beta, gamma)):
        d, b, g = dbg
        i0 = i * n_tot
        i1 = (i+1) * n_tot
        rhs[i0:i1] = d * utils.laplacian_flat_periodic_2d(
                    c[i0:i1], n, L=L, diff_multiplier=diff_multiplier)
        rhs[i0:i1] += b + g * c[i0:i1]

    # Nonlinear terms
    if f is not None:
        rhs += f(c.reshape((n_species, *n)), t, *f_args).flatten()

    return rhs


def rkf45_2d(c0, time_points, n_species, n, D=None, beta=None,
             gamma=None, f=None, f_args=(), L=None, diff_multiplier=None,
             dt0=0.000001, **kw):
    """
    Solve using Runge-Kutta-Fehlberg.
    """

    # Make sure all required kwargs are specified
    if any(x is None for x in
            [D, beta, gamma, f, f_args, L, diff_multiplier]):
        raise RuntimeError('D, beta, gamma, f, f_args, L, '
                           + 'diff_multipler, must all be specified.')

    # Total number of grid points
    n_tot = n[0] * n[1]

    # Do Runge-Kutta-Fehlberg
    args = (n_species, n, D, beta, gamma, f, f_args, L, diff_multiplier)
    c = utils.rkf45(dc_dt, c0.flatten(), time_points, args=args, dt=dt0)

    # Reshape and return
    return c.reshape((n_species, *n, len(time_points)))


def vsimex_2d(c0, time_points, n_species, n, D=None, beta=None, gamma=None,
              f=None, f_args=(), L=None, diff_multiplier=None, k2=None,
              dt0=1e-6, dt_bounds=(0.000001, 100.0), allow_negative=False,
              vsimex_tol=0.001, vsimex_tol_buffer=0.01, k_P=0.075, k_I=0.175,
              k_D=0.01, s_bounds=(0.1, 10.0), quiet=False):
    """
    Does adaptive step size Adams-Bashforth/Crank-Nicholson
    integration.

    s_bounds : 2-tuple of floats
        The bounds on the mutliplier for step size adjustment.
    """

    # Make sure all required kwargs are specified
    if any(x is None for x in
            [D, beta, gamma, f, f_args, L, diff_multiplier, k2]):
        raise RuntimeError('D, beta, gamma, f, f_args, L, '
                           + 'diff_multipler, k2 must all be specified.')

    # Total number of grid points
    n_tot = n[0] * n[1]

    # Tiny concentration
    tiny_conc = 1e-9

    # Do Runge-Kutta-Fehlberg to get to first few time points
    dt = tuple([dt0]*3)
    rkf45_time_points = np.array([time_points[0],
                                  time_points[0] + dt[0],
                                  time_points[0] + dt[0] + dt[1]])
    args = (n_species, n, D, beta, gamma, f, f_args, L, diff_multiplier)
    rkf45_output = utils.rkf45(dc_dt, c0.flatten(), rkf45_time_points,
                               args=args, dt=dt[0]/10)

    # Initialize previous steps from solution
    u = (rkf45_output[:,-2], rkf45_output[:,-1])

    # Initialize the relative change from the time steps
    rel_change = (np.linalg.norm(u[1] - u[0]) / np.linalg.norm(u[1]),
                  np.linalg.norm(u[0] - c0.flatten()) / np.linalg.norm(u[0]))

    # Pull out concentrations and compute FFTs
    c = u[1].reshape((n_species, *n))
    c_hat = np.fft.fftn(c, axes=(1,2))

    # Compute initial f_hat
    f_hat = (np.fft.fft2(
            f(u[0].reshape((n_species, *n)), time_points[0] + dt[0], *f_args)),
                np.fft.fft2(f(c, time_points[0] + dt[0] + dt[1], *f_args)))

    # Set up return variables
    u_sol = [c0.flatten()]
    t = time_points[0] + dt[0] + dt[1]
    t_sol = [time_points[0]]

    # Take the time steps
    while t < time_points[-1]:
        next_time_point_index = np.searchsorted(time_points, t)
        while t < time_points[next_time_point_index]:
            # Compute the CNAB2 step
            c_hat_step = cnab2_step(dt[2], dt[1], c_hat, f_hat, D, beta,
                                    gamma, k2)

            # Convert to real space and build u_step
            c_step = np.fft.ifftn(c_hat_step, axes=(1, 2)).real
            u_step = c_step.flatten()

            # Perform check for negative concentrations
            if not allow_negative:
                u_step, c_step, c_hat_step, reject_step_because_negative = \
                    _check_for_negative_concs(u_step, c_step, c_hat_step, dt,
                                              dt_bounds, tiny_conc=tiny_conc, quiet=quiet)
            else:
                reject_step_because_negative = False

            # Compute the relative change
            rel_change_step = np.linalg.norm(u_step - u[1]) / \
                                                np.linalg.norm(u_step)

            # If relative change less than tolerance, take step
            if not reject_step_because_negative:
                if rel_change_step <= vsimex_tol * (1.0 + vsimex_tol_buffer) \
                        or dt[2] <= dt_bounds[0]:
                    # Take the step
                    c_hat, c, u, t, f_hat = _take_step(
                            c_hat_step, c, u, t, f_hat, c_step, u_step, dt, f,
                            f_args)

                    # Adjust step size
                    dt, rel_change = _adjust_step_size_pid(
                            dt, rel_change, rel_change_step, vsimex_tol, k_P,
                            k_I, k_D, dt_bounds, s_bounds)
                else:
                    # Adjust step size, but do not take step
                    # (step may already have been taken if we had neg. conc.)
                    dt = _adjust_step_size_rejected_step(
                          dt, rel_change_step, vsimex_tol, dt_bounds, s_bounds)

        # If the solution blew up, raise an exception
        if np.isnan(u[1]).any() != 0:
            raise RuntimeError('Solution blew up!')

        # Store outputs
        t_sol.append(t)
        u_sol.append(c.flatten())

    # Interpolate solution
    u_interp = utils.interpolate_solution(
        np.array(u_sol).transpose(), np.array(t_sol), time_points)

    # Return reshaped solution
    return u_interp.reshape((n_species, *n, len(time_points)))


def solve(c0, time_points, D=None, beta=None, gamma=None,
          f=None, f_args=(), L=None, diff_multiplier=None, dt0=1e-6,
          dt_bounds=(0.000001, 100.0), allow_negative=False,
          vsimex_tol=0.001, vsimex_tol_buffer=0.01, k_P=0.075, k_I=0.175,
          k_D=0.01, s_bounds=(0.1, 10.0), quiet=False, solver=vsimex_2d):
    """
    Solve a reaction-diffusion system in two-dimensions.
    """
    # Check and convert inputs
    c0, n_species, n, L, D, beta, gamma, f_args = \
                _check_and_update_inputs(c0, L, D, beta, gamma, f, f_args)

    # Compute square of wave numbers
    kx, ky = utils.wave_numbers_2d(n, L=L)
    k2 = (kx**2 + ky**2)

    # Differencing multiplier for Laplacian
    diff_multiplier = utils.diff_multiplier_periodic_2d(n, order=2)

    # If no nonlinear function
    if f is None:
        f = lambda x, t: np.zeros_like(x)
        f_args = ()

    # Solving using VSIMEX
    return solver(
        c0, time_points, n_species, n, D=D, beta=beta, gamma=gamma,
        f=f, f_args=f_args, L=L, diff_multiplier=diff_multiplier, k2=k2,
        dt0=dt0, dt_bounds=dt_bounds, allow_negative=allow_negative,
        vsimex_tol=vsimex_tol, vsimex_tol_buffer=vsimex_tol_buffer, k_P=k_P,
        k_I=k_I, k_D=k_D, s_bounds=s_bounds, quiet=quiet)


def _take_step(c_hat_step, c, u, t, f_hat, c_step, u_step, dt, f, f_args):
    """
    Update variables in taking the CNAB2 step.
    """
    c_hat = c_hat_step
    c = c_step
    u = (u[1], u_step)
    t += dt[2]
    f_hat = (f_hat[1], np.fft.fft2(f(c, t, *f_args)))

    return c_hat, c, u, t, f_hat


def _adjust_step_size_rejected_step(dt, rel_change_step, vsimex_tol, dt_bounds,
                                    s_bounds):
    """
    Adjust step size for a rejected step.
    """
    mult = vsimex_tol / rel_change_step
    if mult < s_bounds[0]:
        mult = s_bounds[0]

    new_dt = mult * dt[2]

    if new_dt < dt_bounds[0]:
        new_dt = dt_bounds[0]

    return (dt[0], dt[1], new_dt)


def _adjust_step_size_pid(dt, rel_change, rel_change_step, vsimex_tol, k_P,
                          k_I, k_D, dt_bounds, s_bounds):
    """
    Adjust the step size using the PID controller.
    """
    mult = (rel_change[1] / rel_change_step)**k_P \
         * (vsimex_tol / rel_change_step)**k_I \
         * (rel_change[0]**2 / rel_change[1] / rel_change_step)**k_D
    if mult > s_bounds[1]:
        mult = s_bounds[1]
    elif mult < s_bounds[0]:
        mult = s_bounds[0]

    new_dt = mult * dt[2]

    if new_dt > dt_bounds[1]:
        new_dt = dt_bounds[1]
    elif new_dt < dt_bounds[0]:
        new_dt = dt_bounds[0]

    dt = (dt[1], dt[2], new_dt)
    rel_change = (rel_change[1], rel_change_step)

    return dt, rel_change


def _check_for_negative_concs(u_step, c_step, c_hat_step, dt, dt_bounds,
                              tiny_conc=1e-9, quiet=False):
    """
    Check to see is any concentrations went negative.
    """
    if (u_step < 0.0).any():
        if not quiet:
            print('NEGATIVE CONC, ', dt)
        # If we can't reduce step size any more
        if dt[2] <= dt_bounds[0]:
            c_step[np.nonzero(c_step[j] < 0.0)] = tiny_conc
            c_hat_step = np.fft.fftn(c_step, axis=(1, 2))
            u_step = c_step.flatten()
            reject_step = False
            if not quiet:
                print(' NEGATIVE CONC, ZEROED OUT, TAKING TINY STEP ')
        else:  # Cut step size by 2
            dt = (dt[0], dt[1], min(dt[2] / 2.0, dt_bounds[0]))
            reject_step = True
            if not quiet:
                print(' REDUCING STEP SIZE ')
    else:
        reject_step = False

    return u_step, c_step, c_hat_step, reject_step


def cnab2_step(dt_current, dt0, c_hat, f_hat, D, beta, gamma, k2):
    """
    Takes a Crank-Nicolson/Adams-Bashforth (2nd order) step for
    RD system in Fourier space.

    Parameters
    ----------
    dt_current : float
        Current time step
    dt0 : float
        Previous time step
    c_hat : array_like, shape (n_species * total_n_grid_points, )
        Current FFT of concentrations.
    f_hat : array_like, shape (n_species * total_n_grid_points, )
        Current FFT of nonlinear part of dynamics. Organized where
        first len(c_hat)/len(D) entries are flattened array of
        concentrations of first chemical species, next
        len(c_hat)/len(D) for for second chemical species, and
        so on.
    D : array_like, shape (n_species, )
        Array of diffusion coefficients for all chemical species.
    beta : array_like, shape (n_species, )
        Array of autoproduction constants.
    gamma : array_like, shape (n_species, )
        Array of degradation constants. Note that negative gamma
        means degradation.
    k2 : array_like, shape (total_n_grid_points, )
        The square of the wave number for the grid points as a
        flattened array

    Returns
    -------
    output : array_like, shape (n_species * total_n_grid_points, )
        Updated FFT of concentrations.
    """

    omega = dt_current / dt0

    # Nonlinear terms
    c_hat_step = (1 + omega/2) * f_hat[1] - omega/2 * f_hat[0]

    # Linear terms
    for i, dbg in enumerate(zip(D, beta, gamma)):
        d, b, g = dbg
        c_hat_step[i,:,:] += b + (1/dt_current - k2*d/2 + g/2) * c_hat[i,:,:]
        c_hat_step[i,:,:] /= (1/dt_current + k2*d/2 - g/2)

    return c_hat_step


def _check_and_update_inputs(c0, L, D, beta, gamma, f, f_args):
    """
    Check inputs and update parameters for convenient use.
    """

    # Use initial concentrations to get problem dimensions
    c0, n_species, n = _check_and_update_c0(c0)

    # Make sure dimensions make sense.
    n = _check_n_gridpoints(n)

    # Check D, beta, gamma for consistency
    D = _check_beta_gamma_D(D, n_species, name='D')
    beta = _check_beta_gamma_D(beta, n_species, name='beta')
    gamma = _check_beta_gamma_D(gamma, n_species, name='gamma')

    # Perform further checks and updates
    f_args = _check_f(f, f_args)
    L = _check_L(L)

    # At least one of D, beta, gamma, or f must not be None
    if (D == 0.0).all() and (beta == 0.0).all() \
            and (gamma == 0.0).all() and f is None:
        raise RuntimeError(
                'At least one of D, beta, gamma, and f must be nonzero.')

    return c0, n_species, n, L, D, beta, gamma, f_args


def _check_and_update_c0(c0):
    """
    Check c0.
    """

    # Convert to Numpy array in case it's a list of lists or something
    c0 = np.array(c0)

    # Convert to 3D array for single species
    if len(c0.shape) == 2:
        c0 = np.array([c0])

    if len(c0.shape) != 3:
        raise RuntimeError('c0 must be an n_species by nx by ny numpy array')

    # Make sure all concentrations are nonnegative
    if (c0 < 0).any():
        raise RuntimeError('All entries in c0 must be nonnegative.')

    # Determine number of species.
    n_species = c0.shape[0]

    # Extract number of grid points
    n = tuple(c0.shape[1:])

    return c0, n_species, n


def _check_beta_gamma_D(x, n_species, name='beta, gamma, D arrays'):
    if x is None:
        return np.zeros(n_species)

    if np.isscalar(x):
        x = np.array([x])

    # Make sure it's a numpy array
    if type(x) in [list, tuple]:
        D = np.array(x)

    if len(x.shape) != 1:
        raise RuntimeError(f'{name} must be a one-dimensional array.')

    # Make sure arrays have proper length
    if len(x) != n_species:
        raise RuntimeError(f'len({name}) must equal c0.shape[0].')

    # Make sure it is a float
    x = x.astype(float)

    return x.astype(float)


def _check_n_gridpoints(n):
    """
    Check number of grid points meet requirements.
    """
    # Number of grid points must be tuple of ints
    if type(n) in [list, np.ndarray]:
        n = tuple(n)

    if type(n) != tuple or len(n) != 2:
        raise RuntimeError('`n` must be a 2-tuple.')

    # Make sure the number of grid points are ints
    if type(n[0]) != int or type(n[1]) != int:
        raise RuntimeError('Number of grid points must be integer.')

    # Make sure they are even
    if n[0] % 2 != 0 or n[1] % 2 != 0:
        raise RuntimeError('Number of grid points must be even.')

    return n


def _check_f(f, f_args):
    """
    Check arguments for reaction function.
    """
    if f is None and f_args is not None and f_args != ():
        raise RuntimeError('f is None, but f_args are given.')

    if f_args is None or type(f_args) in [list, np.ndarray]:
        f_args = tuple()

    # Check the call signature of f
    if f is not None:
        f_params = inspect.signature(f).parameters

        # Make sure there are no variable keyword args in f; no f(**kw).
        for key in f_params:
            if f_params[key].kind == inspect._ParameterKind.VAR_KEYWORD:
                raise RuntimeError('f cannot accept **kwargs')

        # Make sure we have correct number of args
        if len(f_params) != len(f_args) + 2:
            err_str = 'f must have call signature f(c, t, *args). '
            err_str += 'Length of f_args and f signature mismatch.'
            raise RuntimeError(err_str)

    return f_args


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
        raise RuntimeError('`L` must be a 2-tuple.')

    # Make sure the lengths are float
    return (float(L[0]), float(L[1]))
