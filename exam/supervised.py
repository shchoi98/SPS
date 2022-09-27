import numpy as np
import torch as t
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def fit_wh(X, Y):
    return t.inverse(X.T @ X) @ X.T @ Y


def main():
    # 1D fake data
    # N = 100  # number of datapoints
    # D = 1  # dimension of datapoints
    # sigma = 0.1  # output noise
    # X = t.randn(N, D)
    # Wtrue = t.ones(D, 1)
    # Y = X @ Wtrue + sigma * t.randn(N, 1)
    #
    # fig, ax = plt.subplots()
    # ax.set_xlabel("$x_\lambda$")
    # ax.set_ylabel("$y_\lambda$")
    # ax.scatter(X, Y);
    #
    # Wh = fit_wh(X, Y)
    # print(f"Wtrue = {Wtrue.T}")
    # print(f"Wh    = {Wh.T}")
    #
    # fig, ax = plt.subplots()
    # ax.set_xlabel("$x_\lambda$")
    # ax.set_ylabel("$y_\lambda$")
    # ax.scatter(X, Y, label="data")
    # ax.plot(X, X @ Wh, 'r', label="fitted line")
    # ax.legend();
    # # result
    # # Wtrue = tensor([[1.]])
    # # Wh    = tensor([[1.0021]])

    print(' ')

    # 2D data
    # N = 100  # number of datapoints
    # D = 2  # dimension of datapoints
    # sigma = 0.3  # output noise
    # X = t.randn(N, D)
    # Wtrue = t.ones(D, 1)
    # Y = X @ Wtrue + sigma * t.randn(N, 1)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlabel("$x_{\lambda, 0}$")
    # ax.set_ylabel("$x_{\lambda, 1}$")
    # ax.set_zlabel("$y_{\lambda, 0}$")
    # ax.scatter(xs=X[:, 0], ys=X[:, 1], zs=Y[:, 0]);
    #
    # Wh = fit_wh(X, Y)
    # print(f"Wtrue = {Wtrue.T}")
    # print(f"Wh    = {Wh.T}")
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlabel("$x_{\lambda, 0}$")
    # ax.set_ylabel("$x_{\lambda, 1}$")
    # ax.set_zlabel("$y_{\lambda, 0}$")
    # ax.scatter(X[:, 0], X[:, 1], Y[:, 0])
    #
    # Xp = t.tensor([
    #     [-4., -4.],
    #     [-4., 4.],
    #     [4., -4.],
    #     [4., 4.]
    # ])
    #
    # ax.plot_trisurf(
    #     np.array(Xp[:, 0]),
    #     np.array(Xp[:, 1]),
    #     np.array((Xp @ Wh)[:, 0]),
    #     color='r',
    #     alpha=0.3
    # );
    #
    # # result
    # # Wtrue = tensor([[1., 1.]])
    # # Wh    = tensor([[1.0524, 1.0166]])

    print(' ')

    N = 100  # number of datapoints
    D = 1  # dimension of datapoints
    sigma = 0.1  # output noise
    X = t.rand(N, D)
    Wtrue = 2 * t.ones(D, 1)
    btrue = 3
    Y = X @ Wtrue + btrue + sigma * t.randn(N, 1)

    fig, ax = plt.subplots()
    ax.set_xlabel("$x_\lambda$")
    ax.set_ylabel("$y_\lambda$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 7)
    ax.scatter(X, Y);

    Wh = fit_wh(X, Y)
    print(f"Wtrue = {Wtrue.T}")
    print(f"Wh    = {Wh.T}")

    fig, ax = plt.subplots()
    ax.set_xlabel("$x_\lambda$")
    ax.set_ylabel("$y_\lambda$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 7)
    ax.scatter(X, Y, label="data")
    ax.plot(X, X @ Wh, 'r', label="fitted line")
    ax.legend();


if __name__ == '__main__':
    main()