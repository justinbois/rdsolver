import numba
import numpy as np
import scipy.optimize

def asdm(D_a=0.005, D_s=0.2, rho_a=0.01, rho_s=0.02, sigma_a=0.0, sigma_s=0.02,
         mu_a=0.01, kappa_a=0.25):
    """
    Set up parameters for the activator-substrate depletion model
    (ASDM).

    Notes
    -----
    .. This is the ASDM model as given by equation 3 of Kohn and
       Meinhardt, Rev. Mod. Phys., 1994.
    """
    D = np.array([D_a, D_s])
    beta = np.array([sigma_a, sigma_s])
    gamma = np.array([-mu_a, 0.0])

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

        f1 = omega_de * c_de
        f2 = omega_dD * c_d * c_D
        f3 = omega_E * c_d * c_E
        f4 = omega_eE * c_d * c_E * c_de**2
        f5 = omega_D * c_D

        f_D = f1 - f2
        f_E = f1 - f3 - f4
        f_d = f5 + f2 - f3 - f4
        f_de = f3 + f4

        return np.stack((f_D, f_E, f_d, f_de))

    f_args = (omega_D, omega_E, omega_dD, omega_eE, omega_de)

    beta = np.zeros(4, dtype=np.float)
    gamma = np.array([-omega_D, 0.0, 0.0, -omega_de])

    # Find homogeneous steady state
    def cd_from_cde(cde, ce0, omega_E, omega_eE, omega_de):
        if cde == ce0:
            return 0.0
        else:
            return omega_de * cde / (ce0 - cde)*(omega_E + omega_eE * cde**2)

    def cde_root(cde, cd0, ce0, omega_D, omega_E, omega_dD, omega_eE, omega_de):
        cd = cd_from_cde(cde, ce0, omega_E, omega_eE, omega_de)
        cD = cd0 - cd - cde
        return omega_de * cde - omega_dD * cd * cD - omega_D * cD

    try:
        cde_ss = scipy.optimize.brentq(cde_root, 0, 0.3,
                args=(cd0, ce0, omega_D, omega_E, omega_dD, omega_eE, omega_de))
        cE_ss = ce0 - cde_ss
        cd_ss = cd_from_cde(cde_ss, ce0, omega_E, omega_eE, omega_de)
        cD_ss = cd0 - cde_ss - cd_ss

        return D, beta, gamma, f, f_args, np.array([cD_ss, cE_ss, cd_ss, cde_ss])
    except:
        return D, beta, gamma, f, f_args, None
