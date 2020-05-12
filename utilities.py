import sys
import torch as t
import pandas as pd
import numpy as np
import tensorflow as tf
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
    # x_ones = t.ones(x.shape[0], 1)
    return x_e


def quad(x):
    a = np.ones(x.shape)
    x_square = x ** 2
    x_e = np.column_stack((a, x, x_square))
    # x_2 = t.cat([x ** 2, x, t.ones(x.shape[0], 1)], 1)
    return x_e


def cubic(x):
    a = np.ones(x.shape)
    x_square = x ** 2
    x_cube = x ** 3
    x_e = np.column_stack((a, x, x_square, x_cube))
    return x_e


# def expo(x):
    # return np.exp(x)


def error(x, y, w):

    errors = y - x.dot(w)
    return errors.T.dot(errors)


# calculate the fitting weight
def fit_wh(x, y):
    return np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)


def display(filename):
    xx, yy = load_points_from_file(filename)
    view_data_segments(xx, yy)

# you need to use pytorch to plot or calculate the least square and plot the fit line


def plot_l(x, y, w):
    fig, ax = plt.subplots()

    ax.scatter(x, y, s=100)

    x_min = x.min()
    x_max = x.max()
    y_x_min = w[0] + w[1] * x_min
    y_x_max = w[0] + w[1] * x_max
    ax.plot([x_min, x_max], [y_x_min, y_x_max], 'r-', lw=2)
    plt.show()


def plot_q(x, y, w):
    fig, ax = plt.subplots()

    ax.scatter(x, y, s=100)

    x_min = x.min()
    x_max = x.max()
    xs = t.linspace(x_min, x_max, 100)
    ax.plot(xs, quad(xs).dot(w), 'r')
    plt.show()


def plot_c(x, y, w):
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=100)

    x_min = x.min()
    x_max = x.max()
    xs = t.linspace(x_min, x_max, 100)
    ax.plot(xs, cubic(xs).dot(w), 'r')
    plt.show()


def main():
    file_name = sys.argv[1]
    x, y = load_points_from_file("train_data/"+file_name)

    # xs_e = line(x)
    # xt = t.from_numpy(xs_e)
    # yt = tf.convert_to_tensor(y)

    # 이거는 데이터 자를는 기능
    len_data = len(x)
    num_segments = len_data//20

    # x_e = line(xt)
    # w = fit_wh(x_e, yt)
    # plot(xt, yt, w)

    xs = np.split(x, num_segments, axis=0)
    ys = np.split(y, num_segments, axis=0)
    total_error = 0

    ww = []
    for i in range(num_segments):

        xs_l = line(xs[i])
        xs_q = quad(xs[i])
        xs_c = cubic(xs[i])

        wl = fit_wh(xs_l, ys[i])
        wq = fit_wh(xs_q, ys[i])
        wc = fit_wh(xs_c, ys[i])

        el = error(xs_l, ys[i], wl)
        eq = error(xs_q, ys[i], wq)
        ec = error(xs_c, ys[i], wc)

        e = min(el, eq, ec)
        total_error += e

        if e == el:
            print("linear")
            print(wl)
            ww = np.append(ww, 'wl')
        elif e == eq:
            print("quadratic")
            print(wq)
            ww = np.append(ww, 'wq')
        elif e == ec:
            print("cubic")
            print(wc)
            ww = np.append(ww, 'wc')

    print(total_error)

    if len(sys.argv) == 3 and sys.argv[2] == "--plot":
        fig, ax = plt.subplots()
        ax.scatter(x, y, s=100)

        for i in range(num_segments):

            xr = t.linspace(xs[i].min(), xs[i].max(), 100)

            if ww[i] == 'wl':
                xs_l = line(xs[i])
                wl = fit_wh(xs_l, ys[i])
                ax.plot(xr, line(xr).dot(wl), 'r')
            elif ww[i] == 'wq':
                xs_q = quad(xs[i])
                wq = fit_wh(xs_q, ys[i])
                ax.plot(xr, quad(xr).dot(wq), 'r')
            elif ww[i] == 'wc':
                xs_c = cubic(xs[i])
                wc = fit_wh(xs_c, ys[i])
                ax.plot(xr, cubic(xr).dot(wc), 'r')
        plt.show()


if __name__ == '__main__':
    main()
