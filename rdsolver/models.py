import numba
import numpy as np
import scipy.optimize

def asdm(D_a=0.005, D_s=0.2, rho_a=0.01, rho_s=0.02, sigma_a=0.0, sigma_s=0.02,
         mu_a=0.01, kappa_a=0.25):
    """
    Set up parameters for the activator-substrate depletion model
    (ASDM).

    Parameters
    ----------
    D_a : float
        Diffusion coefficient of activator.
    D_s : float
        Diffusion coefficient of substrate.
    rho_a : float
        Production rate of activator at saturation.
    rho_s : float
        Delpletion rate of inhibitor at saturation.
    sigma_a : float
        Basal production rate of activator.
    sigma_s : float
        Basal production rate of substrate.
    mu_a : float
        Autodegration constant of activator.
    kappa_a : float
        Saturation constant for activator.

    Returns
    -------
    D : ndarray, shape (2,)
        Diffusion coefficients
    beta : ndarray, shape (2,)
        Basal production rates
    gamma : ndarray shape (2, 2)
        Linear coefficients for reactions
    f : Numba'd function
        Function for nonlinear reaction dynamics.
    f_args : 1-tuple
        Arguments for f
    home_ss : ndarray, shape (2,)
        Homogeneous steady states of activator and substrate.

    Notes
    -----
    .. This is the ASDM model as given by equation 3 of Kohn and
       Meinhardt, Rev. Mod. Phys., 1994.
    """
    D = np.array([D_a, D_s])
    beta = np.array([sigma_a, sigma_s])
    gamma = np.diag([-mu_a, 0.0])

    @numba.jit(nopython=True)
    def f(c, t, rho_a, rho_s, kappa_a):
        a = c[0,:,:]
        s = c[1,:,:]
        a2 = a**2
        a2s = a2 * s
        fa = rho_a * a2s / (1 + kappa_a * a2)
        fs = -rho_s * a2s / (1 + kappa_a * a2)
        return np.stack((fa, fs))

    f_args = (rho_a, rho_s, kappa_a)

    a_ss = sigma_a / mu_a + rho_a * sigma_s / rho_s / mu_a
    s_ss = sigma_s / rho_s * (1 + kappa_a * a_ss**2) / a_ss**2

    return D, beta, gamma, f, f_args, np.array([a_ss, s_ss])


def asdm_switch(D_a=0.015, D_s=0.03, D_y=0, rho_a=0.025, rho_s=0.0025,
                rho_y=0.03, sigma_a=0.0, sigma_s=0.00225, sigma_y=0.00015,
                mu_a=0.025, mu_s=0.00075, mu_y=0.003, kappa_a=0.1, kappa_s=20.0,
                kappa_y=22.0):
    """
    Set up parameters for the activator-substrate depletion model
    (ASDM) with a switch.

    Notes
    -----
    .. This is the ASDM model as given by equation 8 of Kohn and
       Meinhardt, Rev. Mod. Phys., 1994.
    """
    D = np.array([D_a, D_s, D_y])

    gamma = np.array([[-mu_a, 0, 0],
                      [0, -mu_s, 0],
                      [sigma_y, 0, -mu_y]], dtype=np.float)

    if kappa_s == 0:
        beta = np.array([sigma_a, sigma_s, 0.0], dtype=np.float)

        @numba.jit(nopython=True)
        def f(c, t, rho_a, rho_s, rho_y, sigma_s, sigma_y, kappa_a, kappa_y):
            a = c[0,:,:]
            s = c[1,:,:]
            y = c[2,:,:]
            a2 = a**2
            y2 = y**2
            a2s = a2 * s
            fa = rho_a * a2s / (1 + kappa_a * a2)
            fs = -rho_s * a2s / (1 + kappa_a * a2)
            fy = rho_y * y2 / (1 + kappa_y * y2)
            return np.stack((fa, fs, fy))

        f_args = (rho_a, rho_s, rho_y, sigma_s, sigma_y, kappa_a, kappa_y)
    else:
        beta = np.array([sigma_a, 0, 0])

        @numba.jit(nopython=True)
        def f(c, t, rho_a, rho_s, rho_y, sigma_s, sigma_y,
              kappa_a, kappa_s, kappa_y):
            a = c[0,:,:]
            s = c[1,:,:]
            y = c[2,:,:]
            a2 = a**2
            y2 = y**2
            a2s = a2 * s
            fa = rho_a * a2s / (1 + kappa_a * a2)
            fs = sigma_s / (1 + kappa_s * y) - rho_s * a2s / (1 + kappa_a * a2)
            fy = rho_y * y2 / (1 + kappa_y * y2)
            return np.stack((fa, fs, fy))

        f_args = (rho_a, rho_s, rho_y, sigma_s, sigma_y,
                  kappa_a, kappa_s, kappa_y)

    return D, beta, gamma, f, f_args, None


def min_system(D_D=60.0, D_E=60.0, D_d=1.2, D_de=0.4, omega_D=2.9e-4,
               omega_E=1.9e-9, omega_dD=4.8e-8, omega_eE=2.1e-20,
               omega_de=0.029, cd0=1, ce0=0.3):
    """
    Set up parameters for the MinD/MinE system.

    Notes
    -----
    .. This model comes from Loose, et al., Science, 2008.
    """

    D = np.array([D_D, D_E, D_d, D_de])

    @numba.jit(nopython=True)
    def f(c, t, omega_D, omega_E, omega_dD, omega_eE, omega_de):
        c_D = c[0,:,:]
        c_E = c[1,:,:]
        c_d = c[2,:,:]
        c_de = c[3,:,:]

        f2 = omega_dD * c_d * c_D
        f3 = omega_E * c_d * c_E
        f4 = omega_eE * c_d * c_E * c_de**2

        f_D = -f2
        f_E = -f3 - f4
        f_d = f2 - f3 - f4
        f_de = f3 + f4

        return np.stack((f_D, f_E, f_d, f_de))

    f_args = (omega_D, omega_E, omega_dD, omega_eE, omega_de)

    beta = np.zeros(4, dtype=np.float)
    gamma = np.array([[-omega_D, 0, 0, omega_de],
                      [0, 0, 0, omega_de],
                      [omega_D, 0, 0, 0],
                      [0, 0, 0, -omega_de]], dtype=np.float)

    return D, beta, gamma, f, f_args, None
