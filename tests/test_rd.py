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


def test_dc_dt():
    # Even concentration field, two species, no rxns
    n = (6, 6)
    D = np.array((1.0, 1.0))
    L = None
    c = np.ones(72)
    t = 0
    assert np.isclose(0, rd.dc_dt(c, t, D, n)).all()

    # No diffusion, uniform concentration field, simple decay
    n = (4, 4)
    D = np.array([1.0])
    L = None
    gamma = rdsolver.rd._update_gamma(-1.0, n)
    c = np.ones(16)
    t = 0
    rxn_args = ()
    assert np.isclose(-c, rd.dc_dt(c, t, D, n, gamma=gamma)).all()

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
    assert np.isclose(correct, rd.dc_dt(c, t, D, n)).all()
