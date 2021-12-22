import collections
import numpy as np
import scipy.io as scio
from matplotlib import pyplot as plt
from sklearn.cluster import kmeans_plusplus
import matplotlib.pyplot as plt
np.random.seed(1234)
data = scio.loadmat('sarcos_inv.mat')['sarcos_inv']
np.random.shuffle(data)
n = data.shape[0]
xs_train = data[:int(n*0.8),:21]
ys_train = data[:int(n*0.8),21].reshape(-1,1)
xs_valid = data[int(n*0.8):,:21]
ys_valid = data[int(n*0.8):,21].reshape(-1,1)
data = scio.loadmat('sarcos_inv_test.mat')['sarcos_inv_test']
xs_test = data[:,:21]
ys_test = data[:,21].reshape(-1,1)

def standardized(data):
    means = data.mean(axis=0)
    stds = data.std(axis=0)
    return (data - means) / stds

xs_train_std = standardized(xs_train)
ys_train_std = standardized(ys_train)
xs_valid_std = standardized(xs_valid)
ys_valid_std = standardized(ys_valid)
xs_test_std = standardized(xs_test)
ys_test_std = standardized(ys_test)

def gaussian_transform(x,x_centers):
    N,D = x.shape
    K,D = x_centers.shape
    re = np.array([]).reshape(-1,K)
    a = 1.0/(np.sqrt(2*np.pi*(25**2)))
    for i in range(N):
        r_2 = ((x_centers-x[i,:])**2).sum(axis=1)
        psi = a*np.exp(-r_2/(2*625))
        re = np.vstack((re,psi))
    return re


def my_variance(xs):
    mean = np.mean(xs)
    sums = 0
    for v in xs:
        sums += (v[0] - mean) ** 2
    return sums / xs.size

def my_smse(z1, z2, s):
    N = z1.size
    sums = 0
    for i in range(N):
        sums += (z1[i] - z2[i]) ** 2
    return sums / (N * s)
var_ys_train = my_variance(ys_train_std)

x = [(i+1)*10 for i in range(10)]
y = []

for i in range(10):
    xs_centers = kmeans_plusplus(xs_train_std, (i+10)*10, random_state=0)[0]
    xs_train_gauss = gaussian_transform(xs_train_std, xs_centers)
    w_glr = np.linalg.inv(xs_train_gauss.T.dot(xs_train_gauss)).dot(xs_train_gauss.T).dot(ys_train_std)
    xs_valid_gauss = gaussian_transform(xs_valid_std, xs_centers)
    ys_pred_gauss_valid = xs_valid_gauss.dot(w_glr)
    smse_gauss = my_smse(ys_pred_gauss_valid, ys_valid_std, var_ys_train)
    y.append(smse_gauss)

plt.figure()
plt(x,y)