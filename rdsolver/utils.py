import numba
import numpy as np
import scipy.interpolate

def grid_points_1d(n, L=None, x_start=0.0):
    """
    Returns the grid points for differencing on a periodic grid in 1D.

    Parameters
    ----------
    n : int
        Number of grid points
    L : float, default 2*pi
        Physical length of grid.
    x_start : float, default 0.0
        Leftmost point of grid

    Returns
    -------
    output : ndarray, shape (n, )
        Positions of grid points

    Notes
    -----
    .. The position at `x_start` is omitted because it is equivalent
       to the last.
    """
    if L is None:
        L = 2 * np.pi

    return L * np.arange(1, n+1) / n + x_start


# ###############
def grid_points_2d(n, L=None, x_start=(0.0, 0.0)):
    """
    Returns the grid points for differencing on a periodic grid in 2D.

    Parameters
    ----------
    n : 2-tuple of ints
        n[0] is the number of rows of differencing grid.
        n[1] is the number of columns of differencing grid.
        Thus, n[0] corresponds to the y-direction and n[1]
        to the x-direction.
    L : 2-tuple of floats, or None (default)
        L[0] is the physical height of the domain (the y-direction).
        L[1] is the physical height of the domain (the y-direction).
        If None, L[0] = L[1] = 2*pi.
    x_start : 2-tuple of floats, default (0.0, 0.0)
        x_start[0] is the point where the y-domain starts
        x_start[1] is the point where the x-domain starts

    Returns
    -------
    x : ndarray, shape (n[1], )
        Positions of grid points in the x-direction.
    y : ndarray, shape (n[0], )
        Positions of grid points in the x-direction.
    xx : ndarray, shape (n[0] * n[1], )
        Positions of x-coordinates in flattened 2D grid.
    yy : ndarray, shape (n[0] * n[1], )
        Positions of y-coordinates in flattened 2D grid.
    x_grid : ndarray, shape(n[1], n[0])
        x-values as meshgrid
    y_grid : ndarray, shape(n[1], n[0])
        y-values as meshgrid

    Notes
    -----
    .. The position at `x_start` is omitted because it is equivalent
       to the last.
    .. Flattening ends up giving lexicographic ordering.  I.e., if there
       are n[1] columns in a grid, then entry i*n_col + j in a flattened
       array corresponds to entry i,j in the unflattened grid.
    """

    if L is None:
        L = (2*np.pi, 2*np.pi)

    x = grid_points_1d(n[0], L[0], x_start[0])
    y = grid_points_1d(n[1], L[1], x_start[1])

    # Make grid
    x_grid, y_grid = np.meshgrid(x, y, indexing='ij')

    # Flatten
    xx = x_grid.flatten()
    yy = y_grid.flatten()

    return x, y, xx, yy, x_grid, y_grid

def wave_numbers_1d(n, L=None):
    """
    Compute the wave numbers corresponding to FFT.

    Parameters
    ----------
    n : int
        Number of grid points
    L : float, default 2*pi
        The physical extent of the domain.

    Returns
    -------
    output : ndarray, shape (n, )
        The wave numbers for the FFT.
    """

    if L is None:
        L = 2.0 * np.pi

    if L is None:
        L = 2.0 * np.pi

    return np.fft.fftfreq(n, L / (2 * np.pi * n))


def wave_numbers_2d(n, L=None):
    """
    Compute the wave numbers corresponding to FFT.

    Parameters
    ----------
    n : 2-tuple of ints
        n[0] is the number of rows of differencing grid.
        n[1] is the number of columns of differencing grid.
    L : 2-tuple of floats, or None (default)
        L[0] is the physical height of the domain (the y-direction).
        L[1] is the physical height of the domain (the y-direction).
        If None, L[0] = L[1] = 2*pi.

    Returns
    -------
    kx : ndarray, shape n
        The x-wave numbers for the appropriate FFT.
    ky : ndarray, shape n
        The y-wave numbers for the appropriate FFT.
    """

    if L is None:
        L = (2*np.pi, 2*np.pi)

    kx = wave_numbers_1d(n[0], L=L[0])
    ky = wave_numbers_1d(n[1], L=L[1])

    return np.meshgrid(kx, ky, indexing='ij')


def diff_multiplier_periodic_1d(n, order=1):
    """
    Compute the array to multiply the FFT to differentiate with
    periodic BCs using spectral differentiation.

    Parameters
    ----------
    n : int
        The length of the 1D input data
    order : int, default 1
        The order of the derivative to compute.

    Returns
    -------
    output : array_like, shape (n,)
        Array to multiply FFT by to perform spectral differentiation.
    """

    # Right now, we can only have an even number of grid points.
    if n % 2 != 0:
        raise RuntimeError('Must have even number of grid points.')

    # Make the ik mask (set wave number N/2 = 0 for order odd order)
    if order % 2 == 0:
        wave_numbers = np.concatenate(
            (np.arange(n//2 + 1), np.arange(-n//2 + 1, 0)))
    else:
        wave_numbers = np.concatenate(
            (np.arange(n//2), (0,), np.arange(-n//2 + 1, 0)))

    # Compute the multiplier to compute the derivative
    return (1j * wave_numbers)**order


def diff_multiplier_periodic_2d(n, order=1):
    """
    Compute the array to multiply the FFT to differentiate with
    periodic BCs using spectral differentiation.

    Parameters
    ----------
    n : 2-tuple of ints
        n[0] is the number of rows of differencing grid.
        n[1] is the number of columns of differencing grid.
    order : int, default 1
        The order of the derivative to compute.

    Returns
    -------
    mult_xx : array_like, shape n
        Array to multiply FFT by to perform spectral differentiation
        in the x-direction.
    mult_yy : array_like, shape n
        Array to multiply FFT by to perform spectral differentiation
        in the y-direction.
    """

    # Compute 1D differencing arrays
    mult_x = diff_multiplier_periodic_1d(n[0], order=order)
    mult_y = diff_multiplier_periodic_1d(n[1], order=order)

    # Reshape for 2D differentiation
    mult_xx = np.reshape(np.tile(mult_x, n[1]), n, order='C')
    mult_yy = np.reshape(np.tile(mult_y, n[0]), n, order='F')

    mult_yy, mult_xx = np.meshgrid(mult_x, mult_y, indexing='ij')

    return mult_xx, mult_yy


def diff_periodic_fft_2d(f, order=1, L=None, real_data=True,
                         diff_multiplier=None):
    """
    Computes the spectral derivative of the 2-D function defined at
    periodic grid points.

    Parameters
    ----------
    f : array_like, shape (nx, ny)
        Array of function values to be used in derivative calculation.
    order : int, default 1
        Order of the derivative to calculate.
    L : 2-tuple of floats, or None
        L[0] is the physical height of the domain (the x-direction).
        L[1] is the physical height of the domain (the y-direction).
        If None, L[0] = L[1] = 2*pi.
    real_data : bool, default True
        If True, assume the function being differentiated is real.
    diff_multiplier : array_like, shape (nx, ny), or None
        Multiplier to use in spectral differentiation. If None,
        this is calculated on an interval of length L = [Lx, Ly].
    Returns the x and y derivatives.

    The standard length of the interval is Lx = Ly = 2*pi.  Uses FFTs
    to compute the derivatives.  If f_hat_multiplier is given, does
    does not compute the array to multiply FFT by to compute
    derivatives and uses the supplied array.  This consitutes a speed
    boost, but the user should take great care that the proper
    f_hat_multipliers are provided.
    """

    if diff_multiplier is None:
        diff_multiplier = diff_multiplier_periodic_2d(f.shape, order=order)

    fft_f = np.fft.fft2(f)
    if real_data:
        Dmf = (np.fft.ifft2(diff_multiplier[1] * fft_f).real,
               np.fft.ifft2(diff_multiplier[0] * fft_f).real)
    else:
        Dmf = (np.fft.ifft2(diff_multiplier[1] * fft_f),
               np.fft.ifft2(diff_multiplier[0] * fft_f))

    if L is not None:
        return Dmf[0] * (2*np.pi/L[0])**order, Dmf[1] * (2*np.pi/L[1])**order

    return Dmf


def laplacian_flat_periodic_2d(f, n, L=None, diff_multiplier=None):
    """
    Compute Laplacian of flattened array.

    Parameters
    ----------
    f : nd_array, shape (n[0]*n[1], )
        Flattened array of concentrations.
    n : 2-tuple of ints
        n[0] is the number of rows of differencing grid.
        n[1] is the number of columns of differencing grid.
    L : 2-tuple of floats, or None (default)
        L[0] is the physical height of the domain (the y-direction).
        L[1] is the physical height of the domain (the y-direction).
        If None, L[0] = L[1] = 2*pi.
    diff_multiplier : array_like, shape n, or None
        Multiplier to use in spectral differentiation. If None,
        this is calculated on an interval of length L = [Lx, Ly].

    Returns
    -------
    output : nd_array, shape(n[0]*n[1], )
        Flattened Laplacian of the concentration.
    """
    dfdx, dfdy = diff_periodic_fft_2d(f.reshape(n), order=2,
                                      L=L, diff_multiplier=diff_multiplier)
    return (dfdx + dfdy).flatten()


def flat_to_3d(c_flat, n_species, n):
    """
    Convert flat array of concentrations to 3D array.

    Parameters
    ----------
    c_flat : array_like, shape (n_species*n[0]*n[1])
        Flattened array of concentrations
    n_species :
    """

def rkf45_step(f, y, t, args, h, tol, s_bounds, h_min):
    """
    Take an RKF45 step from point y at time t to time t +
    dt for dy/dt = f(y, t, args).

    Parameters
    ----------
    f : function, f(y, t, *args)
        Function containing right hand side of ODEs.
    y : array_like
        Value of solution at time t.
    t : float
        time
    args : tuple
        Arguments to be passed to the function f besides y and t.
    h : float
        Time step size
    tol : float
        Tolerance for error in the step
    s_bounds : 2-tuple of floats
        The bounds on the mutliplier for step size adjustment.
    h_min : float
        Minimal step size.

    Returns
    -------
    y : array_like
        Updated y after time step
    t : float
        Updated time after time step
    h : float
        Updated step size.
    """
    k_1 = h * f(y, t, *args)

    y_2 = y + k_1 / 4.0
    k_2 = h * f(y_2, t + h / 4.0, *args)

    y_3 = y + (3.0 * k_1 + 9.0 * k_2) / 32.0
    k_3 = h * f(y_3, t + 3.0 * h / 8.0, *args)

    y_4 = y + (1932.0 * k_1 - 7200.0 * k_2 + 7296.0 * k_3) / 2197.0
    k_4 = h * f(y_4, t + 12.0 * h / 13.0, *args)

    y_5 = y + (8341.0 * k_1 - 32832.0 * k_2 + 29440.0 * k_3 - 845.0 * k_4)\
        / 4104.0
    k_5 = h * f(y_5, t + h, *args)

    y_6 = y + (-6080.0 * k_1 + 41040.0 * k_2 - 28352.0 * k_3 + 9295.0 * k_4
               - 5643.0 * k_5) / 20520.0
    k_6 = h * f(y_6, t + h / 2.0, *args)

    # Calculate error
    error = (np.abs(209 * k_1 - 2252.8 * k_3 - 2197.0 * k_4 + 1504.8 * k_5
                + 2736.0 * k_6) / 75240.0).max()

    # Either don't take a step or use the RK4 step
    if error < tol or h <= h_min:
        y_new = y + (2375.0 * k_1 + 11264.0 * k_3 + 10985 * k_4
                     - 4104.0 * k_5) / 20520.0
        t += h
    else:
        y_new = y

    # Compute scaling for new step size
    if error == 0.0:
        s = s_bounds[1]
    else:
        s = (tol * h / 2.0 / error)**0.25
    if s < s_bounds[0]:
        s = s_bounds[0]
    elif s > s_bounds[1]:
        s = s_bounds[1]

    return y_new, t, max(s * h, h_min)


def rkf45(f, initial_cond, time_points, args=(), dt=None,
          tol=1.0e-7, s_bounds=(0.1, 10.0), h_min=0.0):
    """
    Solve a system of ODEs using explicit Runge-Kutta-Fehlberg 4-5
    time stepping.

    Solves dy/dt = f(y, t, args), like the built-in SciPy function
    odeint.

    If dt is not given, the first time step is given by
    time_points[1] - time_points[0].

    Returns y: array, shape (len(initial_cond), len(time_points)))
            t: array, shape len(time_points)
    """

    # Set up return variables
    t_sol = [time_points[0]]
    t = time_points[0]
    i_max = len(time_points)
    y = [initial_cond]
    y_0 = initial_cond
    i = 1

    if tol is None:
        tol = 1.0e-7

    if dt is None:
        h = time_points[1] - time_points[0]
    else:
        h = dt

    while i < i_max:
        while t < time_points[i]:
            y_0, t, h = rkf45_step(f, y_0, t, args, h, tol, s_bounds, h_min)
        if t > t_sol[-1]:
            y.append(y_0)
            t_sol.append(t)
        i += 1
        if np.isnan(y_0).any():
            raise ValueError('Solution blew up! Try reducing dt.')

    return interpolate_solution(np.array(y).transpose(),
                                np.array(t_sol), time_points)


@numba.jit(nopython=True)
def laplacian_fd(a, hx, hy):
    dx = -2 * np.copy(a)
    dy = -2 * np.copy(a)
    dx[1:,:] += a[:-1,:]
    dx[0,:] += a[-1,:]
    dx[:-1, :] += a[1:,:]
    dx[-1, :] += a[0,:]

    dy[:,1:] += a[:,:-1]
    dy[:,0] += a[:,-1]
    dy[:,:-1] += a[:,1:]
    dy[:,-1] += a[:,0]

    return dx/hx**2 + dy/hy**2


def spectral_integrate_2d(f, L=None):
    """
    Performs 2-D integration of a function f defined on a uniform
    periodic grid.

    Parameters
    ----------
    f : 2D nd_array
        Function values to integrate
    L : 2-tuple of float, default (2*pi, 2*pi)
        The physical extent of the system.

    Returns
    -------
    output : float
        The integral over both x and y.
    """

    # Size of domain
    nx, ny = f.shape

    # Specify lengths if not given
    if L is None:
        L = (2*np.pi, 2*np.pi)

    # Compute integral
    int_x = L[0] / nx * np.sum(f, axis=0)
    return L[1] / ny * np.sum(int_x)


def interpolate_solution(y, t_sol, t):
    """
    Interpolates the solution of a system of ODEs using B-splines.

    Parameters
    ----------
    y : array_like, shape(n_grid_points, len(t_sol))
        The solution to the sytem of ODEs to be interpolated.
    t_sol : array_like
        The time points of the solution.
    t : array_like
        The time points desired from the interpolation.

    Returns
    -------
    output : array_like, shape(n_grid_points, len(t))
        The interpolated solution
    """

    # If we already have the points, just return
    if len(t_sol) <= 3 or (len(t_sol) == len(t) and (t_sol == t).all()):
        return y

    y_real_interp = np.empty((y.shape[0], len(t)))
    for i in range(y.shape[0]):
        # Make B-spline
        tck = scipy.interpolate.splrep(t_sol, y.real[i,:])

        # Evaluate B-spline at desired points
        y_real_interp[i,:] = scipy.interpolate.splev(t, tck)

    if np.iscomplex(y).any():
        y_imag_interp = np.zeros((y.shape[0], len(t)))
        for i in range(y.shape[0]):
            # Make B-spline
            tck = scipy.interpolate.splrep(t_sol, y.imag[i,:])

            # Evaluate B-spline at desired points
            y_imag_interp[i,:] = scipy.interpolate.splev(t, tck)

        return y_real_interp + 1j * y_imag_interp
    else:
        return y_real_interp
