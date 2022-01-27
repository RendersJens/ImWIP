# This example shows a more advanced application: motion estimation and correction in 4D-CT

import numpy as np
import astra
import tomopy
import pylops
from matplotlib import pyplot as plt
import imwip

im_size = 256
translation = [25, -10, 15]
rotation = [0.3, 0.2, 0.1]
shepp = tomopy.shepp3d(im_size).astype(np.float32)

# motion operator
M = lambda rot, trans: imwip.AffineWarpingOperator3D(
    im_shape=(im_size, im_size, im_size),
    rotation=rot,
    translation=trans,
    adjoint_type="inverse"
)

# simulate
x = shepp.ravel()
b = M(rotation, translation) @ x

# solve
# ---------------------

# objective function
def f(rot, trans):
    res = M(rot, trans) @ x - b
    return 1/2 * np.dot(res, res)

# gradient of objective function
def grad_f(rot, trans):
    res = M(rot, trans) @ x - b
    dMx_rot, dMx_trans = imwip.diff(M(rot, trans), x, to=["rot", "trans"])
    grad_rot = dMx_rot.T @ res
    grad_trans = dMx_trans.T @ res
    return grad_rot, grad_trans

# calback function: print motion parameters each iteration
callback = lambda rot, trans: print(rot, trans)

# initial guess: zero
rot0 = [0.0, 0.0, 0.0]
trans0 = [0.0, 0.0, 0.0]
x0 = np.zeros(im_size**3, dtype=np.float32)

# solve
rot, trans = imwip.split_barzilai_borwein(
    grad_f,
    x0=(rot0, trans0),
    max_iter=500,
    verbose=True,
    callback=callback
)

print(rot, trans)