import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold, datasets
from tqdm import tqdm


def TSNE(X, dim=2):
    X = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
    """N, C->N, dim"""
    tsne = manifold.TSNE(n_components=dim, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)
    print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    return X_norm


""" X:feature  shape:H,W,C """
def singel_TSNE(X, save_path):
    X_norm = TSNE(X)
    plt.figure(figsize=(16, 12))

    X_norm = np.swapaxes(X_norm, 0, 1)
    # dim, N
    plt.scatter(X_norm[0], X_norm[1], s=30, color='red')
    # for i in tqdm(range(X_norm.shape[0])):
    #     color = 'red'
    #     alpha = 1.0
    #     s = 10
    #     plt.scatter(X_norm[i, 0], X_norm[i, 1], s=s, color=color, alpha=alpha)

    plt.savefig(save_path)


def singel_3d_TSNE(X, save_path):
    X_norm = TSNE(X, dim=3)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('x', fontsize=10, color='black')
    ax.set_ylabel('y', fontsize=10, color='black')
    ax.set_zlabel('z', fontsize=10, color='black')

    X_norm = np.swapaxes(X_norm, 0, 1)
    # 3, H*W
    ax.scatter(X_norm[0], X_norm[1], X_norm[2], s=10, c='red', alpha=1)
    plt.savefig(save_path)


def dual_TSNE(X1, X2, save_path):
    X1_norm = TSNE(X1)
    X2_norm = TSNE(X2)
    plt.figure(figsize=(16, 12))

    X1_norm = np.swapaxes(X1_norm, 0, 1)
    X2_norm = np.swapaxes(X2_norm, 0, 1)
    plt.scatter(X1_norm[0], X1_norm[1], s=30, alpha=1, color='red')
    plt.scatter(X2_norm[0], X2_norm[1], s=30, alpha=1, color='royalblue')

    plt.savefig(save_path)


def dual_3d_TSNE(X1, X2, save_path):
    X1_norm = TSNE(X1, dim=3)
    X2_norm = TSNE(X2, dim=3)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('x', fontsize=10, color='black')
    ax.set_ylabel('y', fontsize=10, color='black')
    ax.set_zlabel('z', fontsize=10, color='black')

    X1_norm = np.swapaxes(X1_norm, 0, 1)
    X2_norm = np.swapaxes(X2_norm, 0, 1)
    ax.scatter(X1_norm[0], X1_norm[1], X1_norm[2], s=10, c='red', alpha=1)
    ax.scatter(X2_norm[0], X2_norm[1], X2_norm[2], s=10, c='royalblue', alpha=1)

    plt.savefig(save_path)
