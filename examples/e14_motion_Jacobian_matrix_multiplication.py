import numpy as np
import imwip
from time import time

# settings
hr_dim = np.array([216,256,76]).astype(int) #use [32, 32, 32] for testing
backend ='numba' #or numba_cpu
stdmotion = 1.5
ra = np.random.normal(0,stdmotion,3).astype(np.float32)
tr = np.random.normal(0,stdmotion,3).astype(np.float32)
center = hr_dim.astype(np.float32) / 2 - 0.5
M = imwip.AffineWarpingOperator3D(hr_dim, translation=tr, rotation=ra, center=center, backend=backend)
n_cols = 15

# benchmark
rr0 = np.ones(M.shape[1]).astype(np.float32)
t0 = time()
for i in np.arange(n_cols):
    Jr0r, Jr0t = M.derivative(rr0,['rot','trans'])
dt0 = time() - t0
print(dt0)

rr1 = np.ones((M.shape[1],1)).astype(np.float32)
t1 = time()
for i in np.arange(n_cols):
    Jr1r, Jr1t = M.derivative(rr1,['rot','trans'])
dt1 = time() - t1
print(dt1)

rr2 = np.ones([M.shape[1], n_cols]).astype(np.float32)
t2 = time()
Jr2r, Jr2t = M.derivative(rr2,['rot','trans'])
dt2 = time() - t2
print(dt2)

print('Time gain',np.round((dt0-dt2)/dt0*100,2),'%') #12%