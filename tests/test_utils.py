import numpy as np
import pytest

import rdsolver as rd

def test_grid_points_1d():
    # Test standard
    correct = np.array([1, 2, 3, 4, 5]).astype(float) / 5 * 2 * np.pi
    assert np.isclose(rd.utils.grid_points_1d(5), correct).all()

    # Test standard with specified length
    correct = np.array([1, 2, 3, 4, 5]).astype(float)
    assert np.isclose(rd.utils.grid_points_1d(5, L=5), correct).all()

    # Test different starting point
    correct = np.array([1, 2, 3, 4, 5]).astype(float) / 5 * 2 * np.pi - 1.0
    assert np.isclose(rd.utils.grid_points_1d(5, x_start=-1.0), correct).all()


def test_grid_points_2d():
    # Test standard
    n = (5, 5)
    correct_x = np.array([1, 2, 3, 4, 5])  / 5 * 2 * np.pi
    correct_y = np.array([1, 2, 3, 4, 5])  / 5 * 2 * np.pi
    correct_x_grid = np.array([[1, 1, 1, 1, 1],
                               [2, 2, 2, 2, 2],
                               [3, 3, 3, 3, 3],
                               [4, 4, 4, 4, 4],
                               [5, 5, 5, 5, 5]]) / 5 * 2 * np.pi
    correct_y_grid = np.array([[1, 2, 3, 4, 5],
                               [1, 2, 3, 4, 5],
                               [1, 2, 3, 4, 5],
                               [1, 2, 3, 4, 5],
                               [1, 2, 3, 4, 5]]) / 5 * 2 * np.pi
    correct_xx = np.array([1]*5 + [2]*5 + [3]*5 + [4]*5 + [5]*5) / 5 * 2 * np.pi
    correct_yy = np.array([1, 2, 3, 4, 5]*5) / 5 * 2 * np.pi
    x, y, xx, yy, x_grid, y_grid = rd.utils.grid_points_2d(n)
    assert np.isclose(x, correct_x).all()
    assert np.isclose(y, correct_y).all()
    assert np.isclose(x_grid, correct_x_grid).all()
    assert np.isclose(y_grid, correct_y_grid).all()
    assert np.isclose(xx, correct_xx).all()
    assert np.isclose(yy, correct_yy).all()

    # Test standard with different number of grid points
    n = (5, 6)
    correct_x = np.array([1, 2, 3, 4, 5])  / 5 * 2 * np.pi
    correct_y = np.array([1, 2, 3, 4, 5, 6])  / 6 * 2 * np.pi
    correct_x_grid = np.array([[1, 1, 1, 1, 1, 1],
                               [2, 2, 2, 2, 2, 2],
                               [3, 3, 3, 3, 3, 3],
                               [4, 4, 4, 4, 4, 4],
                               [5, 5, 5, 5, 5, 5]]) / 5 * 2 * np.pi
    correct_y_grid = np.array([[1, 2, 3, 4, 5, 6],
                               [1, 2, 3, 4, 5, 6],
                               [1, 2, 3, 4, 5, 6],
                               [1, 2, 3, 4, 5, 6],
                               [1, 2, 3, 4, 5, 6]]) / 6 * 2 * np.pi
    correct_xx = np.array([1]*6 + [2]*6 + [3]*6 + [4]*6 + [5]*6) / 5 * 2 * np.pi
    correct_yy = np.array([1, 2, 3, 4, 5, 6]*5) / 6 * 2 * np.pi
    x, y, xx, yy, x_grid, y_grid = rd.utils.grid_points_2d(n)
    assert np.isclose(x, correct_x).all()
    assert np.isclose(y, correct_y).all()
    assert np.isclose(x_grid, correct_x_grid).all()
    assert np.isclose(y_grid, correct_y_grid).all()
    assert np.isclose(xx, correct_xx).all()
    assert np.isclose(yy, correct_yy).all()

    # Test different physical lengths and different number of grid poitns
    n = (5, 6)
    L = (2*np.pi, 1)
    correct_x = np.array([1, 2, 3, 4, 5]) / 5 * 2 * np.pi
    correct_y = np.array([1, 2, 3, 4, 5, 6]) / 6
    correct_x_grid = np.array([[1, 1, 1, 1, 1, 1],
                               [2, 2, 2, 2, 2, 2],
                               [3, 3, 3, 3, 3, 3],
                               [4, 4, 4, 4, 4, 4],
                               [5, 5, 5, 5, 5, 5]]) / 5 * 2 * np.pi
    correct_y_grid = np.array([[1, 2, 3, 4, 5, 6],
                               [1, 2, 3, 4, 5, 6],
                               [1, 2, 3, 4, 5, 6],
                               [1, 2, 3, 4, 5, 6],
                               [1, 2, 3, 4, 5, 6]]) / 6
    correct_xx = np.array([1]*6 + [2]*6 + [3]*6 + [4]*6 + [5]*6) / 5 * 2 * np.pi
    correct_yy = np.array([1, 2, 3, 4, 5, 6]*5) / 6
    x, y, xx, yy, x_grid, y_grid = rd.utils.grid_points_2d(n, L=L)
    assert np.isclose(x, correct_x).all()
    assert np.isclose(y, correct_y).all()
    assert np.isclose(x_grid, correct_x_grid).all()
    assert np.isclose(y_grid, correct_y_grid).all()
    assert np.isclose(xx, correct_xx).all()
    assert np.isclose(yy, correct_yy).all()

    # Test different physical lengths
    n = (5, 5)
    L = (2*np.pi, 1)
    correct_x = np.array([1, 2, 3, 4, 5])  / 5 * 2 * np.pi
    correct_y = np.array([1, 2, 3, 4, 5])  / 5
    correct_x_grid = np.array([[1, 1, 1, 1, 1],
                               [2, 2, 2, 2, 2],
                               [3, 3, 3, 3, 3],
                               [4, 4, 4, 4, 4],
                               [5, 5, 5, 5, 5]]) / 5 * 2 * np.pi
    correct_y_grid = np.array([[1, 2, 3, 4, 5],
                               [1, 2, 3, 4, 5],
                               [1, 2, 3, 4, 5],
                               [1, 2, 3, 4, 5],
                               [1, 2, 3, 4, 5]]) / 5
    correct_xx = np.array([1]*5 + [2]*5 + [3]*5 + [4]*5 + [5]*5) / 5 * 2 * np.pi
    correct_yy = np.array([1, 2, 3, 4, 5]*5) / 5
    x, y, xx, yy, x_grid, y_grid = rd.utils.grid_points_2d(n, L=L)
    assert np.isclose(x, correct_x).all()
    assert np.isclose(y, correct_y).all()
    assert np.isclose(x_grid, correct_x_grid).all()
    assert np.isclose(y_grid, correct_y_grid).all()
    assert np.isclose(xx, correct_xx).all()
    assert np.isclose(yy, correct_yy).all()


def test_wave_numbers_1d():
    # 2π domain length
    correct = np.array([0, 1, 2, -3, -2, -1])
    assert (correct == rd.utils.wave_numbers_1d(6)).all()

    # Other domain lengths
    L = 1
    correct = np.array([0, 1, 2, -3, -2, -1]) * (2 * np.pi / L)
    assert (correct == rd.utils.wave_numbers_1d(6, L=L)).all()

    L = 7.89
    correct = np.array([0, 1, 2, -3, -2, -1]) * (2 * np.pi / L)
    assert (correct == rd.utils.wave_numbers_1d(6, L=L)).all()

    # Odd domains
    correct = np.array([0, 1, 2, 3, -3, -2, -1])
    assert (correct == rd.utils.wave_numbers_1d(7)).all()

    L = 1
    correct = np.array([0, 1, 2, 3, -3, -2, -1]) * (2 * np.pi / L)
    assert (correct == rd.utils.wave_numbers_1d(7, L=L)).all()

    L = 7.89
    correct = np.array([0, 1, 2, 3, -3, -2, -1]) * (2 * np.pi / L)
    assert (correct == rd.utils.wave_numbers_1d(7, L=L)).all()


def test_wave_numbers_2d():
    # 2π domain length
    correct_x = np.reshape(np.array([0, 1, 2, -3, -2, -1]*6), (6, 6), order='F')
    correct_y = np.reshape(np.array([0, 1, 2, -3, -2, -1]*6), (6, 6), order='C')
    kx, ky = rd.utils.wave_numbers_2d((6, 6))
    assert (correct_x == kx).all()
    assert (correct_y == ky).all()

    # Mixed number of grid points
    correct_x = np.reshape(np.array([0, 1, 2, -3, -2, -1]*8), (6, 8), order='F')
    correct_y = np.reshape(np.array([0, 1, 2, 3, -4, -3, -2, -1]*6), (6, 8),
                           order='C')
    kx, ky = rd.utils.wave_numbers_2d((6, 8))
    assert (correct_x == kx).all()
    assert (correct_y == ky).all()

    # Mixed number of grid points amd different lengths
    L = (3.4, 5.7)
    correct_x = np.reshape(np.array([0, 1, 2, -3, -2, -1]*8), (6, 8),
                           order='F') * (2*np.pi / L[0])
    correct_y = np.reshape(np.array([0, 1, 2, 3, -4, -3, -2, -1]*6), (6, 8),
                           order='C') * (2*np.pi / L[1])
    kx, ky = rd.utils.wave_numbers_2d((6, 8), L=L)
    assert (correct_x == kx).all()
    assert (correct_y == ky).all()


def test_spectral_integrate_2d():
    L = (2*np.pi, 2*np.pi)
    n = (64, 64)
    x, y, xx, yy, x_grid, y_grid = rd.utils.grid_points_2d(n, L=L)
    f = np.exp(np.sin(x_grid) * np.cos(y_grid))
    correct = 44.649967131680145266
    assert np.isclose(rd.utils.spectral_integrate_2d(f, L=L), correct)

    L = (2*np.pi, 2*np.pi)
    n = (64, 128)
    x, y, xx, yy, x_grid, y_grid = rd.utils.grid_points_2d(n, L=L)
    f = np.exp(np.sin(x_grid) * np.cos(y_grid))
    correct = 44.649967131680145266
    assert np.isclose(rd.utils.spectral_integrate_2d(f, L=L), correct)

    L = (2*np.pi, 4*np.pi)
    n = (128, 64)
    x, y, xx, yy, x_grid, y_grid = rd.utils.grid_points_2d(n, L=L)
    f = np.exp(np.sin(x_grid) * np.cos(y_grid))
    correct = 89.299934263360290533
    assert np.isclose(rd.utils.spectral_integrate_2d(f, L=L), correct)


def test_diff_multiplier_periodic_1d():
    # Error out on odd number of grid points
    with pytest.raises(RuntimeError) as excinfo:
        rd.utils.diff_multiplier_periodic_1d(65)
    excinfo.match('Must have even number of grid points.')

    # First derivative
    correct = np.array([0, 1, 2, 3, 4, 0, -4, -3, -2, -1]) * 1j
    assert (rd.utils.diff_multiplier_periodic_1d(10) == correct).all()

    # Second derivative
    correct = -np.array([0, 1, 2, 3, 4, 5, -4, -3, -2, -1])**2
    assert (rd.utils.diff_multiplier_periodic_1d(10, order=2) == correct).all()

    # Third derivative
    correct = -np.array([0, 1, 2, 3, 4, 0, -4, -3, -2, -1])**3 * 1j
    assert (rd.utils.diff_multiplier_periodic_1d(10, order=3) == correct).all()


def test_diff_multiplier_periodic_2d():
    # Error out on odd number of grid points
    with pytest.raises(RuntimeError) as excinfo:
        rd.utils.diff_multiplier_periodic_2d((65, 64))
    excinfo.match('Must have even number of grid points.')

    # First derivative
    n = (10, 10)
    correct_yy = np.array(
            [[i]*10 for i in [0, 1, 2, 3, 4, 0, -4, -3, -2, -1]]) * 1j
    correct_xx = np.array(
            [[0, 1, 2, 3, 4, 0, -4, -3, -2, -1] for _ in range(10)]) * 1j
    mult_xx, mult_yy = rd.utils.diff_multiplier_periodic_2d(n)
    assert np.isclose(mult_xx, correct_xx).all()
    assert np.isclose(mult_yy, correct_yy).all()

    # Second derivative
    n = (10, 10)
    correct_yy = -np.array(
            [[i]*10 for i in [0, 1, 2, 3, 4, 5, -4, -3, -2, -1]])**2
    correct_xx = -np.array(
            [[0, 1, 2, 3, 4, 5, -4, -3, -2, -1] for _ in range(10)])**2
    mult_xx, mult_yy = rd.utils.diff_multiplier_periodic_2d(n, order=2)
    assert np.isclose(mult_xx, correct_xx).all()
    assert np.isclose(mult_yy, correct_yy).all()


def test_diff_periodic_fft_2d():
    # Test standard grid spacing
    n = (64, 64)
    L = None
    x, y, xx, yy, x_grid, y_grid = rd.utils.grid_points_2d(n, L=L)
    f = np.exp(np.sin(x_grid) * np.cos(y_grid))
    df_dx, df_dy = rd.utils.diff_periodic_fft_2d(f, L=L)
    df_dx_correct = f * np.cos(x_grid) * np.cos(y_grid)
    df_dy_correct = -f * np.sin(x_grid) * np.sin(y_grid)
    assert np.isclose(df_dx, df_dx_correct).all()
    assert np.isclose(df_dy, df_dy_correct).all()

    # Different physical lengths of x and y
    n = (64, 64)
    L = (2*np.pi, 4*np.pi)
    x, y, xx, yy, x_grid, y_grid = rd.utils.grid_points_2d(n, L=L)
    f = np.exp(np.sin(x_grid) * np.cos(y_grid))
    df_dx, df_dy = rd.utils.diff_periodic_fft_2d(f, L=L)
    df_dx_correct = f * np.cos(x_grid) * np.cos(y_grid)
    df_dy_correct = -f * np.sin(x_grid) * np.sin(y_grid)
    assert np.isclose(df_dx, df_dx_correct).all()
    assert np.isclose(df_dy, df_dy_correct).all()

    # Different number of grid points in x and y
    n = (64, 128)
    L = None
    x, y, xx, yy, x_grid, y_grid = rd.utils.grid_points_2d(n, L=L)
    f = np.exp(np.sin(x_grid) * np.cos(y_grid))
    df_dx, df_dy = rd.utils.diff_periodic_fft_2d(f, L=L)
    df_dx_correct = f * np.cos(x_grid) * np.cos(y_grid)
    df_dy_correct = -f * np.sin(x_grid) * np.sin(y_grid)
    assert np.isclose(df_dx, df_dx_correct).all()
    assert np.isclose(df_dy, df_dy_correct).all()

    # Different number of grid points in x and y and different lengths
    n = (64, 128)
    L = (4*np.pi, 2*np.pi)
    x, y, xx, yy, x_grid, y_grid = rd.utils.grid_points_2d(n, L=L)
    f = np.exp(np.sin(x_grid) * np.cos(y_grid))
    df_dx, df_dy = rd.utils.diff_periodic_fft_2d(f, L=L)
    df_dx_correct = f * np.cos(x_grid) * np.cos(y_grid)
    df_dy_correct = -f * np.sin(x_grid) * np.sin(y_grid)
    assert np.isclose(df_dx, df_dx_correct).all()
    assert np.isclose(df_dy, df_dy_correct).all()

    # Test standard grid spacing, second derivative
    n = (64, 64)
    L = None
    x, y, xx, yy, x_grid, y_grid = rd.utils.grid_points_2d(n, L=L)
    f = np.exp(np.sin(x_grid) * np.cos(y_grid))
    df_dx, df_dy = rd.utils.diff_periodic_fft_2d(f, L=L, order=2)
    df_dx_correct = f * np.cos(y_grid) \
            * (np.cos(x_grid)**2 * np.cos(y_grid) - np.sin(x_grid))
    df_dy_correct = f * np.sin(x_grid) \
            * (np.sin(y_grid)**2 * np.sin(x_grid) - np.cos(y_grid))
    assert np.isclose(df_dx, df_dx_correct).all()
    assert np.isclose(df_dy, df_dy_correct).all()

    # Different physical lengths of x and y, second derivative
    n = (64, 64)
    L = (2*np.pi, 4*np.pi)
    x, y, xx, yy, x_grid, y_grid = rd.utils.grid_points_2d(n, L=L)
    f = np.exp(np.sin(x_grid) * np.cos(y_grid))
    df_dx, df_dy = rd.utils.diff_periodic_fft_2d(f, L=L, order=2)
    df_dx_correct = f * np.cos(y_grid) \
            * (np.cos(x_grid)**2 * np.cos(y_grid) - np.sin(x_grid))
    df_dy_correct = f * np.sin(x_grid) \
            * (np.sin(y_grid)**2 * np.sin(x_grid) - np.cos(y_grid))
    assert np.isclose(df_dx, df_dx_correct).all()
    assert np.isclose(df_dy, df_dy_correct).all()

    # Different number of grid points in x and y, second derivative
    n = (64, 128)
    L = None
    x, y, xx, yy, x_grid, y_grid = rd.utils.grid_points_2d(n, L=L)
    f = np.exp(np.sin(x_grid) * np.cos(y_grid))
    df_dx, df_dy = rd.utils.diff_periodic_fft_2d(f, L=L, order=2)
    df_dx_correct = f * np.cos(y_grid) \
            * (np.cos(x_grid)**2 * np.cos(y_grid) - np.sin(x_grid))
    df_dy_correct = f * np.sin(x_grid) \
            * (np.sin(y_grid)**2 * np.sin(x_grid) - np.cos(y_grid))
    assert np.isclose(df_dx, df_dx_correct).all()
    assert np.isclose(df_dy, df_dy_correct).all()

    # Different number of grid points in x and y and diff len, second derivative
    n = (64, 128)
    L = (4*np.pi, 2*np.pi)
    x, y, xx, yy, x_grid, y_grid = rd.utils.grid_points_2d(n, L=L)
    f = np.exp(np.sin(x_grid) * np.cos(y_grid))
    df_dx, df_dy = rd.utils.diff_periodic_fft_2d(f, L=L, order=2)
    df_dx_correct = f * np.cos(y_grid) \
            * (np.cos(x_grid)**2 * np.cos(y_grid) - np.sin(x_grid))
    df_dy_correct = f * np.sin(x_grid) \
            * (np.sin(y_grid)**2 * np.sin(x_grid) - np.cos(y_grid))
    assert np.isclose(df_dx, df_dx_correct).all()
    assert np.isclose(df_dy, df_dy_correct).all()

def test_laplacian_flat_periodic_2d():
    # Same shape in x and y, standard grid
    n = (64, 64)
    L = None
    x, y, xx, yy, x_grid, y_grid = rd.utils.grid_points_2d(n, L=L)
    f = np.exp(np.sin(xx) * np.cos(yy))
    correct = f * np.cos(yy) * (np.cos(xx)**2 * np.cos(yy) - np.sin(xx)) \
            + f * np.sin(xx) * (np.sin(yy)**2 * np.sin(xx) - np.cos(yy))
    assert np.isclose(correct, rd.utils.laplacian_flat_periodic_2d(f, n)).all()
