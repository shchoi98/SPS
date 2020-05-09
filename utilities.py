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
    x_i = t.cat(x, t.ones(x.shape[0]), 1)
    return x_i


def poly(x, order):
    x_i = x
    if order == 2:
        x_i = t.cat([x ** 2, x, t.ones(x.shape[0], 1)], 1)

    elif order == 3:
        x_i = t.cat([x**3, x**2, x, t.ones(x.shape[0], 1)], 1)
    return x_i

# def expo(x):
#     return math.exp(x)


def error(x, y, w):
    errors = y-x.dot(w)
    return errors.T.dot(errors)


# calculate the fitting weight
def fit_wh(x, y):
    return np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)


def display(filename):
    xx, yy = load_points_from_file(filename)
    view_data_segments(xx, yy)

# you need to use pytorch to plot or calculate the least square and plot the fit line


def main():
    file_name = sys.argv[1]
    x, y = load_points_from_file("train_data/"+file_name)

    x.shape = (x.shape[0], 1)
    y.shape = (y.shape[0], 1)

    a = np.ones(x.shape)
    x_e = np.column_stack((a, x))
    w = fit_wh(x_e, y)

    x_min = x.min()
    x_max = x.max()
    y_x_min = w[0] + w[1] * x_min
    y_x_max = w[0] + w[1] * x_max

    if len(sys.argv) == 3 and sys.argv[2] == "--plot":
        fig, ax = plt.subplots()
        ax.scatter(x, y, s=100)
        ax.plot([x_min, x_max], [y_x_min, y_x_max], 'r-', lw=2)
        plt.show()

    e = error(x_e, y, w)
    print(e[0][0])

    # plot

    # print(file_name)
    # print(x,y)


    #     view_data_segments(x, y)
    # #         and plot regression line as well


if __name__ == '__main__':
    main()
