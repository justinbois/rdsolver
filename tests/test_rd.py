import numpy as np
import pytest

import rdsolver as rd
import rdsolver.rd

def test_update_beta():
    # Only one
    n = (10, 9)
    correct = np.array([2.0]*90)
    assert (rdsolver.rd._update_beta(2.0, n) == correct).all()

    # Only one, but as array
    n = (10, 9)
    correct = np.array([2.0]*90)
    assert (rdsolver.rd._update_beta(np.array([2.0]), n) == correct).all()

    # Two, but as array
    n = (10, 9)
    correct = np.array([2.0]*90 + [3.0]*90)
    assert (rdsolver.rd._update_beta(np.array([2.0, 3.0]), n) == correct).all()

    # Two, but as tuple
    n = (10, 9)
    correct = np.array([2.0]*90 + [3.0]*90)
    assert (rdsolver.rd._update_beta((2, 3), n) == correct).all()


def test_update_gammaa():
    # Only one
    n = (10, 9)
    correct = np.array([2.0]*90)
    assert (rdsolver.rd._update_gamma(2.0, n) == correct).all()

    # Only one, but as array
    n = (10, 9)
    correct = np.array([2.0]*90)
    assert (rdsolver.rd._update_gamma(np.array([2.0]), n) == correct).all()

    # Two, but as array
    n = (10, 9)
    correct = np.array([2.0]*90 + [3.0]*90)
    assert (rdsolver.rd._update_gamma(np.array([2.0, 3.0]), n) == correct).all()

    # Two, but as tuple
    n = (10, 9)
    correct = np.array([2.0]*90 + [3.0]*90)
    assert (rdsolver.rd._update_gamma((2, 3), n) == correct).all()

def test_dc_dt():
    # Even concentration field, two species, no rxns
    n = (6, 6)
    D = np.array((1.0, 1.0))
    L = None
    c = np.ones(72)
    t = 0
    assert np.isclose(0, rd.dc_dt(c, t, D, 1, n)).all()

    # No diffusion, uniform concentration field, simple decay
    n = (4, 4)
    D = np.array([1.0])
    L = None
    gamma = rdsolver.rd._update_gamma(-1.0, n)
    c = np.ones(16)
    t = 0
    rxn_args = ()
    assert np.isclose(-c, rd.dc_dt(c, t, D, 1, n, gamma=gamma)).all()

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
    assert np.isclose(correct, rd.dc_dt(c, t, D, 1, n)).all()

    # Simple decay, nonuniform concentration field
    n = (64, 64)
    D = np.array([1.0])
    L = None
    gamma = rdsolver.rd._update_gamma(-1.0, n)
    t = 0
    x, y, xx, yy, x_grid, y_grid = rd.utils.grid_points_2d(n, L=L)
    c = np.exp(np.sin(xx) * np.cos(yy))
    laplacian = c * np.cos(yy) * (np.cos(xx)**2 * np.cos(yy) - np.sin(xx)) \
              + c * np.sin(xx) * (np.sin(yy)**2 * np.sin(xx) - np.cos(yy))
    correct = D[0] * laplacian - c
    assert np.isclose(correct, rd.dc_dt(c, t, D, 1, n, gamma=gamma)).all()

    # Simple decay, nonuniform concentration field, different # of grid points
    n = (64, 128)
    D = np.array([1.0])
    L = None
    gamma = rdsolver.rd._update_gamma(-1.0, n)
    t = 0
    x, y, xx, yy, x_grid, y_grid = rd.utils.grid_points_2d(n, L=L)
    c = np.exp(np.sin(xx) * np.cos(yy))
    laplacian = c * np.cos(yy) * (np.cos(xx)**2 * np.cos(yy) - np.sin(xx)) \
              + c * np.sin(xx) * (np.sin(yy)**2 * np.sin(xx) - np.cos(yy))
    correct = D[0] * laplacian - c
    assert np.isclose(correct, rd.dc_dt(c, t, D, 1, n, gamma=gamma)).all()

    # decay+prod., nonuniform concentration field, different # of grid points
    n = (64, 128)
    D = np.array([1.0])
    L = None
    gamma = rdsolver.rd._update_gamma(-1.0, n)
    beta = rdsolver.rd._update_beta(1.4, n)
    t = 0
    x, y, xx, yy, x_grid, y_grid = rd.utils.grid_points_2d(n, L=L)
    c = np.exp(np.sin(xx) * np.cos(yy))
    laplacian = c * np.cos(yy) * (np.cos(xx)**2 * np.cos(yy) - np.sin(xx)) \
              + c * np.sin(xx) * (np.sin(yy)**2 * np.sin(xx) - np.cos(yy))
    correct = D[0] * laplacian - c + 1.4
    assert np.isclose(correct,
                      rd.dc_dt(c, t, D, 1, n, gamma=gamma, beta=beta)).all()


    # Inclusion of nonlinear terms
    n = (64, 128)
    D = np.array([1.0])
    L = None
    gamma = rdsolver.rd._update_gamma(-1.0, n)
    beta = rdsolver.rd._update_beta(1.4, n)
    def f(c, t):
        return -c**2
    t = 0
    x, y, xx, yy, x_grid, y_grid = rd.utils.grid_points_2d(n, L=L)
    c = np.exp(np.sin(xx) * np.cos(yy))
    laplacian = c * np.cos(yy) * (np.cos(xx)**2 * np.cos(yy) - np.sin(xx)) \
              + c * np.sin(xx) * (np.sin(yy)**2 * np.sin(xx) - np.cos(yy))
    correct = D[0] * laplacian - c + 1.4 - c**2
    assert np.isclose(correct,
                     rd.dc_dt(c, t, D, 1, n, gamma=gamma, beta=beta, f=f)).all()

    # Inclusion of nonlinear terms, multiple species
    n = (64, 128)
    D = np.array([1.0, 2.0])
    f_args = (2.4, 5.6)
    L = None
    gamma = rdsolver.rd._update_gamma((-1.0, -0.5), n)
    beta = rdsolver.rd._update_beta((1.4, 1.0), n)
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
                      rd.dc_dt(c, t, D, 2, n, gamma=gamma, beta=beta, f=f,
                      f_args=f_args)).all()
