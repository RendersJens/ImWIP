import numpy as np
import imwip
from time import time

# settings
hr_d = 256
hr_dim = np.array([hr_d,hr_d,hr_d]).astype(int)
stdmotion = 1.5
np.random.seed(0)
ra = np.random.normal(0,stdmotion,3).astype(np.float32)
tr = np.random.normal(0,stdmotion,3).astype(np.float32)
center = hr_dim.astype(np.float32) / 2.0 - 0.5
n_cols = 15

''''benchmark'''
# multiple backends
#backends =('cpp','numba','numba_cpu')
#for backend in backends: #cpp or numba_cpu
# single backend
backend = 'numba_cpu'
M = imwip.AffineWarpingOperator3D(hr_dim, translation=tr, rotation=ra, center=center, backend=backend)

print(backend+'_mode0')
rr0 = np.ones(M.shape[1]).astype(np.float32)
t0 = time()
for i in np.arange(n_cols):
    Jr0r, Jr0t = M.derivative(rr0,['rot','trans'])
dt0 = time() - t0
dt0_eff = np.mean(M.derivative_time)*n_cols
print('total time',dt0)
print('effective time',dt0_eff)
M.reset_timer()

print(backend+'_mode1')
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
dt1_eff = np.mean(M.derivative_time)*n_cols
print('total time',dt1)
print('effective time',dt1_eff)
M.reset_timer()

print(backend+'_mode2')
rr2 = np.ones((M.shape[1],n_cols)).astype(np.float32)
t2 = time()
Jr2r, Jr2t = M.derivative(rr2,['rot','trans'])
dt2 = time() - t2
dt2_eff = np.mean(M.derivative_time)
print('total time',dt2)
print('effective time',dt2_eff)
M.reset_timer()


