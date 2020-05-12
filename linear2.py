# import sys
# import pandas as pd
# import numpy as np
# from matplotlib import pyplot as plt
#
#
# def load_points_from_file(filename):
#     """Loads 2d points from a csv called filename
#     Args:
#         filename : Path to .csv file
#     Returns:
#         (xs, ys) where xs and ys are a numpy array of the co-ordinates.
#     """
#     points = pd.read_csv(filename, header=None)
#     return points[0].values, points[1].values
#
#
# def error(x, y, w):
#     errors = y-x.dot(w)
#     return errors.T.dot(errors)
#
#
# # calculate the fitting weight
# def fit_wh(x, y):
#     return np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
#
#
# def line(x):
#     a = np.ones(x.shape)
#     x_e = np.column_stack((a, x))
#     return x_e
#
#
# def main():
#     file_name = sys.argv[1]
#     x, y = load_points_from_file("train_data/" + file_name)
#
#     x.shape = (x.shape[0], 1)
#     y.shape = (y.shape[0], 1)
#
#     x_e = line(x)
#     w = fit_wh(x_e, y)
#
#     x_min = x.min()
#     x_max = x.max()
#     y_x_min = w[0] + w[1] * x_min
#     y_x_max = w[0] + w[1] * x_max
#
#     e = error(x_e, y, w)
#     print(e[0][0])
#
#     if len(sys.argv) == 3 and sys.argv[2] == "--plot":
#         fig, ax = plt.subplots()
#         ax.scatter(x, y, s=100)
#         ax.plot([x_min, x_max], [y_x_min, y_x_max], 'r-', lw=2)
#         plt.show()
#
