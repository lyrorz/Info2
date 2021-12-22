import collections

import numpy as np
import scipy.io as scio
from matplotlib import pyplot as plt
np.random.seed(1234)
n = np.random.random(1)
print(n)
x = np.array([[1,2],[3,4]])
print(x.sum(axis=1))