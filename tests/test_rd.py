import numpy as np
import pytest

import rdsolver as rd
import rdsolver.rd


def test_check_beta_gamma_D():
    correct = np.array([2.0])
    assert np.isclose(correct,
                      rdsolver.rd._check_beta_gamma_D(correct, 1)).all()

    correct = np.array([2.0])
    assert np.isclose(2,
                      rdsolver.rd._check_beta_gamma_D(correct, 1)).all()

    correct = np.array([2.0])
    assert np.isclose(2.0,
                      rdsolver.rd._check_beta_gamma_D(correct, 1)).all()

    correct = np.array([2., 3., 4.])
    assert np.isclose(correct, rdsolver.rd._check_beta_gamma_D(correct, 3)).all()

    with pytest.raises(RuntimeError) as excinfo:
        rdsolver.rd._check_beta_gamma_D(2.0, 2, name='x')
    excinfo.match('len\(x\) must equal c0.shape\[0\].')

    with pytest.raises(RuntimeError) as excinfo:
        rdsolver.rd._check_beta_gamma_D(np.ones((2, 2)), 2, name='x')
    excinfo.match('x must be a one-dimensional array.')


def test_check_f():
    with pytest.raises(RuntimeError) as excinfo:
        rdsolver.rd._check_f(None, (2.5,))
    excinfo.match('f is None, but f_args are given.')

    assert rdsolver.rd._check_f(None, ()) == ()
    assert rdsolver.rd._check_f(None, None) == ()
    assert rdsolver.rd._check_f(lambda x, t: x, ()) == ()
    assert rdsolver.rd._check_f(lambda x, t: x, None) == ()


def test_check_and_update_inputs():
    # Single species, 2D array
    c0 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    correct_c0 = c0
    D = 1.0
    L = None
    beta = None
    gamma = None
    f = None
    f_args = ()
    c0, n_species, n, L, D, beta, gamma, f_args = \
        rdsolver.rd._check_and_update_inputs(c0, L, D, beta, gamma, f, f_args)
    assert (correct_c0 == c0).all()
    assert n_species == 1
    assert n == (2, 4)
    assert type(L) == tuple \
                and np.isclose(np.array(L), 2 * np.pi * np.ones(2)).all()
    assert (D == np.array([1.0])).all()
    assert (beta == np.array([0.0])).all()
    assert (gamma == np.array([0.0])).all()
    assert f_args == ()

    # Two species
    c0_0 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    c0_1 = np.array([[9, 10, 11, 12], [13, 14, 15, 16]])
    c0 = np.stack((c0_0, c0_1))
    correct_c0 = c0
    D = np.array([1.0, 0.0])
    L = None
    beta = None
    gamma = None
    f = None
    f_args = ()
    c0, n_species, n, L, D, beta, gamma, f_args = \
        rdsolver.rd._check_and_update_inputs(c0, L, D, beta, gamma, f, f_args)
    assert (correct_c0 == c0).all()
    assert n_species == 2
    assert n == (2, 4)
    assert type(L) == tuple \
                and np.isclose(np.array(L), 2 * np.pi * np.ones(2)).all()
    assert (D == np.array([1.0, 0.0])).all()
    assert (beta == np.array([0.0, 0.0])).all()
    assert (gamma == np.array([0.0, 0.0])).all()
    assert f_args == ()

    c0_0 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    c0_1 = np.array([[9, 10, 11, 12], [13, 14, 15, 16]])
    c0 = np.stack((c0_0, c0_1))
    correct_c0 = c0
    D = None
    L = None
    beta = None
    gamma = None
    f = None
    f_args = ()
    with pytest.raises(RuntimeError) as excinfo:
        rdsolver.rd._check_and_update_inputs(c0, L, D, beta, gamma, f, f_args)
        excinfo.match('At least one of D, beta, gamma, and f must be nonzero.')


def test_dc_dt():
    # Even concentration field, two species, no rxns
    n = (6, 6)
    D = np.array((1.0, 1.0))
    L = None
    c = np.ones(72)
    t = 0
    assert np.isclose(0, rd.dc_dt(c, t, 2, n, D)).all()

    # No diffusion, uniform concentration field, simple decay
    n = (4, 4)
    D = np.array([1.0])
    L = None
    gamma = np.array([-1.0])
    c = np.ones(16)
    t = 0
    rxn_args = ()
    assert np.isclose(-c, rd.dc_dt(c, t, 1, n, D, gamma=gamma)).all()

    # No reactions, nonuniform concentration field
    n = (64, 64)
    D = np.array([1.0])
    L = None
    t = 0
    x, y, xx, yy, x_grid, y_grid = rd.utils.grid_points_2d(n, L=L)
    c = np.exp(np.sin(xx) * np.cos(yy))
    laplacian = c * np.cos(yy) * (np.cos(xx)**2 * np.cos(yy) - np.sin(xx)) \
              + c * np.sin(xx) * (np.sin(yy)**2 * np.sin(xx) - np.cos(yy))
    correct = D[0] * laplacian
    assert np.isclose(correct, rd.dc_dt(c, t, 1, n, D)).all()

    # Simple decay, nonuniform concentration field
    n = (64, 64)
    D = np.array([1.0])
    L = None
    gamma = np.array([-1.0])
    t = 0
    x, y, xx, yy, x_grid, y_grid = rd.utils.grid_points_2d(n, L=L)
    c = np.exp(np.sin(xx) * np.cos(yy))
    laplacian = c * np.cos(yy) * (np.cos(xx)**2 * np.cos(yy) - np.sin(xx)) \
              + c * np.sin(xx) * (np.sin(yy)**2 * np.sin(xx) - np.cos(yy))
    correct = D[0] * laplacian - c
    assert np.isclose(correct, rd.dc_dt(c, t, 1, n, D, gamma=gamma)).all()

    # Simple decay, nonuniform concentration field, different # of grid points
    n = (64, 128)
    D = np.array([1.0])
    L = None
    gamma = np.array([-1.0])
    t = 0
    x, y, xx, yy, x_grid, y_grid = rd.utils.grid_points_2d(n, L=L)
    c = np.exp(np.sin(xx) * np.cos(yy))
    laplacian = c * np.cos(yy) * (np.cos(xx)**2 * np.cos(yy) - np.sin(xx)) \
              + c * np.sin(xx) * (np.sin(yy)**2 * np.sin(xx) - np.cos(yy))
    correct = D[0] * laplacian - c
    assert np.isclose(correct, rd.dc_dt(c, t, 1, n, D, gamma=gamma)).all()

    # decay+prod., nonuniform concentration field, different # of grid points
    n = (64, 128)
    D = np.array([1.0])
    L = None
    gamma = np.array([-1.0])
    beta = np.array([1.4])
    t = 0
    x, y, xx, yy, x_grid, y_grid = rd.utils.grid_points_2d(n, L=L)
    c = np.exp(np.sin(xx) * np.cos(yy))
    laplacian = c * np.cos(yy) * (np.cos(xx)**2 * np.cos(yy) - np.sin(xx)) \
              + c * np.sin(xx) * (np.sin(yy)**2 * np.sin(xx) - np.cos(yy))
    correct = D[0] * laplacian - c + 1.4
    assert np.isclose(correct,
                      rd.dc_dt(c, t, 1, n, D, gamma=gamma, beta=beta)).all()


    # Inclusion of nonlinear terms
    n = (64, 128)
    D = np.array([1.0])
    L = None
    gamma = np.array([-1.0])
    beta = np.array([1.4])
    def f(c, t):
        return -c**2
    t = 0
    x, y, xx, yy, x_grid, y_grid = rd.utils.grid_points_2d(n, L=L)
    c = np.exp(np.sin(xx) * np.cos(yy))
    laplacian = c * np.cos(yy) * (np.cos(xx)**2 * np.cos(yy) - np.sin(xx)) \
              + c * np.sin(xx) * (np.sin(yy)**2 * np.sin(xx) - np.cos(yy))
    correct = D[0] * laplacian - c + 1.4 - c**2
    assert np.isclose(correct,
                     rd.dc_dt(c, t, 1, n, D, gamma=gamma, beta=beta, f=f)).all()

    # Inclusion of nonlinear terms, multiple species
    n = (64, 128)
    D = np.array([1.0, 2.0])
    f_args = (2.4, 5.6)
    L = None
    gamma = np.array([-1.0, -0.5])
    beta = np.array([1.4, 1.0])
    def f(c, t, k1, k2):
        a, b = c
        return np.array([-k1 * a * b, k2 * a**2 * b])
    t = 0
    n_tot = n[0] * n[1]
    x, y, xx, yy, x_grid, y_grid = rd.utils.grid_points_2d(n, L=L)
    c = np.concatenate((np.exp(np.sin(xx) * np.cos(yy)),
                        np.exp(np.cos(xx) * np.sin(yy))))
    a = c[:n_tot]
    b = c[n_tot:]
    laplacian_0 = a * np.cos(yy) * (np.cos(xx)**2 * np.cos(yy) - np.sin(xx)) \
                + a * np.sin(xx) * (np.sin(yy)**2 * np.sin(xx) - np.cos(yy))
    laplacian_1 = b * np.sin(yy) * (np.sin(xx)**2 * np.sin(yy) - np.cos(xx)) \
                + b * np.cos(xx) * (np.cos(yy)**2 * np.cos(xx) - np.sin(yy))
    correct_a = D[0] * laplacian_0 - a + 1.4 - 2.4 * a * b
    correct_b = D[1] * laplacian_1 - 0.5 * b + 1.0 + 5.6 * a**2 * b
    correct = np.concatenate((correct_a, correct_b))
    assert np.isclose(correct,
                      rd.dc_dt(c, t, 2, n, D, gamma=gamma, beta=beta, f=f,
                      f_args=f_args)).all()

def check_cnab2_expressions(c_hat_step, dt_current, dt0, c_hat, f_hat, D, beta,
                            gamma, k2):
    """
    Check that standard CNAB2 expression holds.
    """
    lhs = (c_hat_step - c_hat) / dt_current

    omega = dt_current / dt0
    # Nonlinear terms
    rhs = (1 + omega/2) * f_hat[1] - omega/2 * f_hat[0]

    # Linear terms
    for i, dbg in enumerate(zip(D, beta, gamma)):
        d, b, g = dbg
        rhs[i,:,:] += b
        rhs[i,:,:] += (g - k2 * d) * (c_hat_step[i,:,:] + c_hat[i,:,:]) / 2

    assert np.isclose(lhs, rhs).all()


def test_cnab2_step():
    dt_current = 1
    dt0 = 1
    n = (64, 64)
    c = (np.ones((1,) + n), np.ones((1,) + n))
    c_hat = np.fft.fftn(c[1], axes=(1, 2))
    f = lambda x, t: x
    f_hat = tuple([np.fft.fftn(f(c_val, 0), axes=(1, 2)) for c_val in c])
    D = np.array([0.0])
    beta = np.array([0.0])
    gamma = np.array([0.0])
    kx, ky = rd.utils.wave_numbers_2d(n)
    k2 = (kx**2 + ky**2)
    c_hat_step = rd.cnab2_step(dt_current, dt0, c_hat, f_hat, D, beta, gamma,
                               k2)
    check_cnab2_expressions(c_hat_step, dt_current, dt0, c_hat, f_hat, D, beta,
                                gamma, k2)

    dt_current = 1
    dt0 = 1.3
    n = (64, 128)
    c = (np.ones((1,) + n), np.ones((1,) + n))
    c_hat = np.fft.fftn(c[1], axes=(1, 2))
    f = lambda x, t: x
    f_hat = tuple([np.fft.fftn(f(c_val, 0), axes=(1, 2)) for c_val in c])
    D = np.array([0.0])
    beta = np.array([0.0])
    gamma = np.array([0.0])
    kx, ky = rd.utils.wave_numbers_2d(n)
    k2 = (kx**2 + ky**2)
    c_hat_step = rd.cnab2_step(dt_current, dt0, c_hat, f_hat, D, beta, gamma,
                               k2)
    check_cnab2_expressions(c_hat_step, dt_current, dt0, c_hat, f_hat, D, beta,
                                gamma, k2)

    dt_current = 1
    dt0 = 1.3
    n = (64, 128)
    c = (np.ones((2,) + n), np.ones((2,) + n))
    c_hat = np.fft.fftn(c[1], axes=(1, 2))
    f = lambda x, t: x
    f_hat = tuple([np.fft.fftn(f(c_val, 0), axes=(1, 2)) for c_val in c])
    D = np.array([0.0, 0.0])
    beta = np.array([0.0, 0.0])
    gamma = np.array([0.0, 0.0])
    kx, ky = rd.utils.wave_numbers_2d(n)
    k2 = (kx**2 + ky**2)
    c_hat_step = rd.cnab2_step(dt_current, dt0, c_hat, f_hat, D, beta, gamma,
                               k2)
    check_cnab2_expressions(c_hat_step, dt_current, dt0, c_hat, f_hat, D, beta,
                                gamma, k2)

    dt_current = 1
    dt0 = 1.3
    n = (64, 128)
    c = (np.ones((2,) + n), np.ones((2,) + n))
    c_hat = np.fft.fftn(c[1], axes=(1, 2))
    f = lambda x, t: x
    f_hat = tuple([np.fft.fftn(f(c_val, 0), axes=(1, 2)) for c_val in c])
    D = np.array([1.0, 0.2])
    beta = np.array([0.6, 10.0])
    gamma = np.array([-1.0, -0.5])
    kx, ky = rd.utils.wave_numbers_2d(n)
    k2 = (kx**2 + ky**2)
    c_hat_step = rd.cnab2_step(dt_current, dt0, c_hat, f_hat, D, beta, gamma,
                               k2)
    check_cnab2_expressions(c_hat_step, dt_current, dt0, c_hat, f_hat, D, beta,
                                gamma, k2)
