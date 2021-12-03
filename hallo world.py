import collections

import numpy as np
import scipy.io as scio
from matplotlib import pyplot as plt


def my_kmeans(xs, init_centers, n_iter):
    """Runs the K-Means algorithm from a given initialization

    Arguments
    xs            2d numpy array of shape (N,D) containing N samples of dimension D
    init_centers  2d numpy array of shape (K,D) containing the initial cluster centers
    n_iter        Number of iterations of the K-Means algorithm

    Returns
    An (K,D) numpy array containing the final cluster centers
    """
    # Your implementation
    N, D = xs.shape
    K, D = init_centers.shape
    centers = init_centers
    clusters = collections.defaultdict(list)
    for i in range(n_iter):
        for j in range(N):
            dis_min = float("inf")
            for k in range(K):
                dis = np.linalg.norm(centers[k, :] - xs[j, :])
                if dis < dis_min:
                    dis_min = dis
                    cluster = k
            clusters[cluster].append(j)
        for j in range(K):
            temp = np.zeros((1, D))
            for v in clusters[j]:
                temp += xs[v, :]
            centers[j] = temp/len(clusters[j])
            print(temp,len(clusters[j]),centers[j])
    return centers
    pass

xs_cluster_test = np.vstack((np.random.normal(loc=(-2,2),scale=np.sqrt(0.2),size=(30,2)),
                            np.random.normal(loc=(-2,-2),scale=np.sqrt(0.2),size=(20,2)),
                            np.random.normal(loc=(2,-1),scale=np.sqrt(0.5),size=(40,2)),
                            np.random.normal(loc=(2,2),scale=np.sqrt(0.5),size=(10,2))))
np.random.shuffle(xs_cluster_test)
init_centers = np.array([[0.0,2.0],[0.0,-2.0]])
centers = my_kmeans(xs_cluster_test,init_centers,1)
plt.scatter(xs_cluster_test[:,0],xs_cluster_test[:,1])
plt.scatter(centers[:,0],centers[:,1],facecolors='red')
plt.show()