import os
import sys

import math
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
    x_i = t.cat(x, x, t.ones(x.shape[0], 1), 1)
    return x_i


def poly(x, order):
    if order == 2:
        t.cat([x ** 2, x, t.ones(x.shape[0], 1)], 1)
    elif order == 3:
        t.cat([x**3, x**2, x, t.ones(x.shape[0], 1)], 1)


def expo(x):
    return expo(x)


def error(x, y, w):
    error_squared= (y - x@w)**2
    return


# calculate the fitting weight
def fit_wh(x, y):
    return t.inverse(x.T @ x) @ x.T @ y


def display(filename):
    xx, yy = load_points_from_file(filename)
    view_data_segments(xx, yy)

# you need to use pytorch to plot or calculate the least square and plot the fit line

# input 
def main():
    file_name = sys.argv[1]
    x, y = load_points_from_file(file_name)

    # for each 20 points loop through

    # Xe = t.cat([x, t.ones(x.shape[0], 1)], 1)
    Wh = fit_wh(x, y)
    # print the summing the error for every line segment
    if len(sys.argv) == 3:
        view_data_segments(x, y)
#         and plot regression line as well


if __name__ == '__main__':
    main()
