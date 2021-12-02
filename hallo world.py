import numpy as np
import scipy.io as scio
print("Hallo world!")

x = scio.loadmat('sarcos_inv.mat')
print(x)
data = scio.loadmat('sarcos_inv.mat')['sarcos_inv']
print(data)
(np.random.shuffle(data))
print(data.shape)
xs_train = data[:][:21]
x = [[1,2,3],[3,4,5],[6,7,8]]
print(x)