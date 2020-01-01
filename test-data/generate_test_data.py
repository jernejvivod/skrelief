import numpy as np
import scipy.io as sio

data = np.random.rand(100, 100)
target = (data[:, 0] > data[:, 2]).astype(int)

sio.savemat('data2.mat', {'data' : data})
sio.savemat('target2.mat', {'target' : target})

