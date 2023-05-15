import numpy as np
import imwip
from time import time

# settings
hr_dim = np.array([64,64,64]).astype(int)
backend ='numba' #cpp or numba_cpu
stdmotion = 1.5
np.random.seed(0)
ra = np.random.normal(0,stdmotion,3).astype(np.float32)
tr = np.random.normal(0,stdmotion,3).astype(np.float32)
center = hr_dim.astype(np.float32) / 2.0 - 0.5
M = imwip.AffineWarpingOperator3D(hr_dim, translation=tr, rotation=ra, center=center, backend=backend)
n_cols = 2

# benchmark

rr0 = np.ones(M.shape[1]).astype(np.float32)
t0 = time()
for i in np.arange(n_cols):
    Jr0r, Jr0t = M.derivative(rr0,['rot','trans'])
dt0 = time() - t0
print(dt0)
print(np.mean(Jr0r))

rr1 = np.ones((M.shape[1],1)).astype(np.float32)
t1 = time()
Jr1r= []
Jr1t= []
for i in np.arange(n_cols):
    J = M.derivative(rr1,['rot','trans'])
    Jr1r.append(J[0])
    Jr1t.append(J[1])
Jr1r = np.array(Jr1r)
Jr1t= np.array(Jr1t)
dt1 = time() - t1
print(dt1)
print(np.mean(Jr1r))

rr2 = np.ones([M.shape[1], n_cols]).astype(np.float32)
t2 = time()
Jr2r, Jr2t = M.derivative(rr2,['rot','trans'])
dt2 = time() - t2
print(dt2)
print(np.mean(Jr2r))

print('Time gain',np.round((dt1-dt2)/dt1*100,2),'%') #12%
