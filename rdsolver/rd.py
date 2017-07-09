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


def _check_and_update_inputs(c0, L, D, beta, gamma, f_args):
    """
    Check inputs and update parameters for convenient use.
    """

    # Use initial concentrations to get problem dimensions
    c0, n_species, n = _check_and_update_c0(c0)

    # Make sure beta, gamma, and D are all arrays or None
    if np.isscalar(D):
        D = np.array([D])

    if np.isscalar(beta):
        beta = np.array([beta])

    if np.isscalar(gamma):
        gamma = np.array([gamma])

    # Make sure arrays have proper length
    if D is not None and len(D) != n_species:
        raise RuntimeError('len(D) must equal c0.shape[0].')

    if beta is not None and len(beta) != n_species:
        raise RuntimeError('len(beta) must equal c0.shape[0].')

    if gamma is not None and len(gamma) != n_species:
        raise RuntimeError('len(gamma) must equal c0.shape[0].')

    # Perform further checks and updates
    n = _check_n_gridpoints(n)
    D = _check_D(D)
    beta = _update_beta(beta, n)
    gamma = _update_gamma(gamma, n)
    f_args = _check_f_args(f_args)
    L = _check_L(L)

    return c0, n_species, n, L, D, beta, gamma, f_args


def _check_and_update_c0(c0):
    """
    Check c0 and convert to flattened array.
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

    # Flatten c0
    c0 = c0.flatten()

    return c0, n_species, n


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

    # Convert to array of floats
    if np.isscalar(beta):
        beta = [beta]
    beta = np.array(beta, dtype=float)

    return np.concatenate([np.array([beta_i] * n[0]*n[1]) for beta_i in beta])


def _update_gamma(gamma, n):
    """
    Convert gamma to something convenient multiply flattened
    concentrations by.

    Parameters
    ----------
    gamma : nd_array or None
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

    return f_args


def _check_D(D):
    """
    Check diffusion coefficient.
    """

    if D is None:
        return None

    if np.isscalar(D):
        D = np.array([D])

    # Make sure it's a numpy array
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
        raise RuntimeError('`n` must be a 2-tuple.')

    # Make sure the number of grid points are ints
    if type(n[0]) != int or type(n[1]) != int:
        raise RuntimeError('Number of grid points must be integer.')

    # Make sure they are even
    if n[0] % 2 != 0 or n[1] % 2 != 0:
        raise RuntimeError('Number of grid points must be even.')

    return n


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
    D : array_like
        Array of diffusion coefficients for species.
    n : 2-tuple of ints
        n[0] is the number of rows of differencing grid.
        n[1] is the number of columns of differencing grid.
    beta : array_like, shape is same as c

    f : function
        Function to compute the nonlinear terms of the dynamics.
        Call signature f(c, t, *f_args), where c[0] is concentration
        of species 0, c[1] is concentration of species 2, etc.
    """
    rhs = np.zeros_like(c)
    n_tot = n[0] * n[1]

    # Compute diffusive terms
    if D is not None:
        for i, d in enumerate(D):
            i0 = i * n_tot
            i1 = (i+1) * n_tot
            rhs[i0:i1] += d * utils.laplacian_flat_periodic_2d(
                        c[i0:i1], n, L=L, diff_multiplier=diff_multiplier)

    # Add reaction terms
    if beta is not None:
        rhs += beta

    if gamma is not None:
        rhs += gamma * c

    if f is not None:
        rhs += f(c.reshape((n_species, n_tot)), t, *f_args).flatten()

    return rhs


def solve(c0, t, L=None, D=None, beta=None, gamma=None, f=None, f_args=(),
          quiet=False):
    """
    Solve a reaction-diffusion system in two-dimensions.
    """
    # Check and convert inputs
    c0, n_species, n, L, D, beta, gamma, f_args = \
                _check_and_update_inputs(c0, L, D, beta, gamma, f_args)

    # Compute square of wave numbers
    kx, ky = utils.wave_numbers_2d(n, L=L)
    k2 = (kx**2 + ky**2).flatten()

    # Differencing multiplier for Laplacian
    diff_multiplier = diff_multiplier_periodic_2d(n, order=2)

    # Solving using VSIMEX
    return vsimex_2d(
        time_points, n_species, n, D=D, beta=beta, gamma=gamma,
        f=f, f_args=f_args, L=L, diff_multiplier=diff_multiplier, dt0=1e-6,
        quiet=quiet)


def vsimex_2d(c0, time_points, n_species, n, D=None, beta=None, gamma=None,
              f=None, f_args=(), L=None, diff_multiplier=None, dt0=1e-6,
              dt_bounds=(0.000001, 100.0), allow_negative=False,
              vsimex_tol=0.001, vsimex_tol_buffer=0.01, k_P=0.075, k_I=0.175,
              k_D=0.01, s_bounds=(0.1, 10.0), quiet=False):
    """
    Does adaptive step size Adams-Bashforth/Crank-Nicholson
    integration.

    s_bounds : 2-tuple of floats
        The bounds on the mutliplier for step size adjustment.
    """

    # Total number of grid points
    n_tot = n[0] * n[1]

    # Tiny concentration
    tiny_conc = 1e-9

    # Do Runge-Kutta-Fehlberg to get to first few time points
    dt = tuple([dt0]*3)
    rkf45_time_points = np.array([time_points[0],
                                  time_points[0] + dt[0],
                                  time_points[0] + dt[0] + dt[1]])
    args = (n_species, n, D, beta, gamma, f, f_args=(), L, diff_multiplier)
    rkf45_output = utils.rkf45(dc_dt, c0, rkf45_time_points, args=args,
                               dt=dt[0]/10)

    # Initialize previous steps from solution
    u = (rkf45_output[:,-2], rkf45_output[:,-1])

    # Initialize the relative change from the time steps
    rel_change = (np.linalg.norm(u[1] - u[0]) / np.linalg.norm(u[1]),
                  np.linalg.norm(u[0] - c0) / np.linalg.norm(u[0]))

    # Pull out concentrations and compute FFTs
    c = tuple(u_entry.reshape((n_species, *n)) for u_entry in u)
    c_hat = tuple(np.fft.fftn(c_entry, axes=(1,2)) for c_entry in c)

    # Compute initial f_hat
    f_hat = (np.fft.fft2(f(c[0], time_points[0] + dt[0], *f_args)),
             np.fft.fft2(f(c[1], time_points[0] + dt[0] + dt[1], *f_args)))

    # Set up return variables
    u_sol = [c0]
    t = time_points[0] + dt[0] + dt[1]
    t_sol = [time_points[0]]
    omega = 1.0

    # Take the time steps
    while t < time_points[-1]:
        next_time_point_index = np.searchsorted(time_points, t)
        while t < time_points[next_time_point_index]:
            omega = dt[2] / dt[1]

            # THIS IS WHERE THE CNAB2 STEP IS
            c_hat_step = cnab2_step()

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
                        c_hat, c, u, t, f_hat, c_step, u_step, dt, f, f_args)

                    # Adjust step size
                    dt, rel_change = _adjust_step_size_pid(
                            dt, rel_change, rel_change_step, vsimex_tol, k_P,
                            k_I, k_D, s_bounds)
                else:
                    # Adjust step size, but do not take step
                    # (step may already have been taken if we had neg. conc.)
                    dt = _adjust_step_size_rejected_step(dt, rel_change_step,
                                                         vsimex_tol, s_bounds)

        # If the solution blew up, raise an exception
        if np.isnan(u[1]).any() != 0:
            raise RuntimeError('Solution blew up!')

        # Store outputs
        t_sol.append(t)
        u_sol.append(c.flatten())

    # Interpolate solution
    return utils.interpolate_solution(
        np.array(u_sol).transpose(), np.array(t_sol), time_points)


def _take_step(c_hat, c, u, t, f_hat, c_step, u_step, dt, f, f_args):
    """
    Update variables in taking the CNAB2 step.
    """
    c_hat = (c_hat[1], c_hat_step)
    c = (c[1], c_step)
    u = (u[1], u_step)
    t += dt[2]
    f_hat = (f_hat[1], np.fft.fft2(f(c[1], t, *f_args)))

    return c_hat, c, u, t, f_hat


def _adjust_step_size_rejected_step(dt, rel_change_step, vsimex_tol, s_bounds):
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
                          k_I, k_D, s_bounds):
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
