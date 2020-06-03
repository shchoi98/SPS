import sys
import torch as t
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt


def load_points_from_file(filename):
    """Loads 2d points from a csv called filename
    Args:
        filename : Path to .csv file
    Returns:
        (xs, ys) where xs and ys are a numpy array of the co-ordinates.
    """
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values


def view_data_segments(xs, ys):
    """Visualises the input file with each segment plotted in a different colour.
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        None
    """
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c=colour)
    plt.show()


# pass x vector and y vector to calculate fit weight
def line(x):
    a = np.ones(x.shape)
    x_e = np.column_stack((a, x))
    return x_e


def quad(x):
    a = np.ones(x.shape)
    x_square = x ** 2
    x_e = np.column_stack((a, x, x_square))
    return x_e


def cubic(x):
    a = np.ones(x.shape)
    x_square = x ** 2
    x_cube = x ** 3
    x_e = np.column_stack((a, x, x_square, x_cube))
    return x_e


def quartic(x):
    a = np.ones(x.shape)
    x_square = x ** 2
    x_cube = x ** 3
    x_4 = x ** 4
    x_e = np.column_stack((a, x, x_square, x_cube, x_4))
    return x_e


def error(x, y, w):
    errors = y - x.dot(w)
    return errors.T.dot(errors)


# calculate the fitting weight
def fit_wh(x, y):
    return np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)


def main():
    file_name = sys.argv[1]
    x, y = load_points_from_file("train_data/"+file_name)

    len_data = len(x)
    num_segments = len_data//20

    xs = np.split(x, num_segments, axis=0)
    ys = np.split(y, num_segments, axis=0)
    total_error = 0

    ww = []
    for i in range(num_segments):

        xs_l = line(xs[i])
        xs_q = quad(xs[i])
        xs_c = cubic(xs[i])
        xs_4 = quartic(xs[i])

        wl = fit_wh(xs_l, ys[i])
        wq = fit_wh(xs_q, ys[i])
        wc = fit_wh(xs_c, ys[i])
        w4 = fit_wh(xs_4, ys[i])

        el = error(xs_l, ys[i], wl)
        eq = error(xs_q, ys[i], wq)
        ec = error(xs_c, ys[i], wc)
        e4 = error(xs_4, ys[i], w4)

        e = min(el, eq, ec, e4)
        total_error += e
        print(e)
        if e == el:
            ww = np.append(ww, 'l')
            print('l')
        elif e == eq:
            ww = np.append(ww, 'q')
            print('q')
        elif e == ec:
            ww = np.append(ww, 'c')
            print('c')
        elif e == e4:
            ww = np.append(ww, '4')
            print('4')

    print(total_error)

    if len(sys.argv) == 3 and sys.argv[2] == "--plot":
        fig, ax = plt.subplots()
        ax.scatter(x, y, s=100)

        for i in range(num_segments):

            xr = t.linspace(xs[i].min(), xs[i].max(), 100)

            if ww[i] == 'l':
                xs_l = line(xs[i])
                wl = fit_wh(xs_l, ys[i])

                ax.plot(xr, line(xr).dot(wl), 'r')

            elif ww[i] == 'q':
                xs_q = quad(xs[i])
                wq = fit_wh(xs_q, ys[i])

                ax.plot(xr, quad(xr).dot(wq), 'r')

            elif ww[i] == 'c':
                xs_c = cubic(xs[i])
                wc = fit_wh(xs_c, ys[i])

                ax.plot(xr, cubic(xr).dot(wc), 'r')

            elif ww[i] == '4':
                xs_4 = quartic(xs[i])
                w4 = fit_wh(xs_4, ys[i])

                ax.plot(xr, quartic(xr).dot(w4), 'r')

        plt.show()


if __name__ == '__main__':
    main()
