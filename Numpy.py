# # import sys
# # import torch as t
# # import pandas as pd
# # import numpy as np
# # from matplotlib import pyplot as plt
# #
# #
# # def load_points_from_file(filename):
# #     """Loads 2d points from a csv called filename
# #     Args:
# #         filename : Path to .csv file
# #     Returns:
# #         (xs, ys) where xs and ys are a numpy array of the co-ordinates.
# #     """
# #     points = pd.read_csv(filename, header=None)
# #     return points[0].values, points[1].values
# #
# #
# # def view_data_segments(xs, ys):
# #     """Visualises the input file with each segment plotted in a different colour.
# #     Args:
# #         xs : List/array-like of x co-ordinates.
# #         ys : List/array-like of y co-ordinates.
# #     Returns:
# #         None
# #     """
# #     assert len(xs) == len(ys)
# #     assert len(xs) % 20 == 0
# #     len_data = len(xs)
# #     num_segments = len_data // 20
# #     colour = np.concatenate([[i] * 20 for i in range(num_segments)])
# #     plt.set_cmap('Dark2')
# #     plt.scatter(xs, ys, c=colour)
# #     plt.show()
# #
# #
# # # pass x vector and y vector to calculate fit weight
# # def line(x):
# #     x_i = t.cat(x, t.ones(x.shape[0]), 1)
# #     return x_i
# #
# #
# # def poly(x):
# #     x_2 = t.cat([x ** 2, x, t.ones(x.shape[0], 1)], 1)
# #     return x_2
# #
# # # def expo(x):
# # #     return math.exp(x)
# #
# #
# # def error(x, y, w):
# #     # x.shape = (x.shape[0], 1)
# #     # y.shape = (y.shape[0], 1)
# #     #
# #     # a = np.ones(x.shape)
# #     # x_e = np.column_stack((a, x))
# #
# #     errors = y-x.dot(w)
# #     return errors.T.dot(errors)
# #
# #
# # # calculate the fitting weight
# # def fit_wh(x, y):
# #     # x.shape = (x.shape[0], 1)
# #     # y.shape = (y.shape[0], 1)
# #     #
# #     # a = np.ones(x.shape)
# #     # x_e = np.column_stack((a, x))
# #
# #     return np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
# #
# #
# # def display(filename):
# #     xx, yy = load_points_from_file(filename)
# #     view_data_segments(xx, yy)
# #
# # # you need to use pytorch to plot or calculate the least square and plot the fit line
# #
# #
# # def plot(x, y, w):
# #     fig, ax = plt.subplots()
# #
# #     ax.scatter(x, y, s=100)
# #
# #     x_min = x.min()
# #     x_max = x.max()
# #     y_x_min = w[0] + w[1] * x_min
# #     y_x_max = w[0] + w[1] * x_max
# #     ax.plot([x_min, x_max], [y_x_min, y_x_max], 'r-', lw=2)
# #     plt.show()
# #
# #
# # def main():
# #     file_name = sys.argv[1]
# #     x, y = load_points_from_file("train_data/"+file_name)
# #
# #     # 이거는 데이터 자를는 기능
# #     len_data = len(x)
# #     num_segments = len_data//20
# #     xs = np.split(x, num_segments, axis=0)
# #     ys = np.split(y, num_segments, axis=0)
# #     total_error = 0
# #     xx = np.empty(20)
# #
# #     for i in range(num_segments):
# #
#         a = np.ones(xs[i].shape)
# #
# #         print()
# #         xs_e = np.column_stack((a, xs[i]))
# #         # print(xs_e)
# #         print("================")
# #         w = fit_wh(xs_e, ys[i])
# #         e = error(xs_e, ys[i], w)
# #
# #         total_error += e
# #         print(w)
# #         if len(sys.argv) == 3 and sys.argv[2] == "--plot":
# #             plot(xs[i], ys[i], w)
# #     print(total_error)
# #
# #
# # if __name__ == '__main__':
# #     main()
