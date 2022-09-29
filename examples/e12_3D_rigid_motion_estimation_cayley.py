import numpy as np
import astra
import tomopy
import pylops
from matplotlib import pyplot as plt
import imwip

im_size = 256
true_trans = [25, -10, 15]
true_rot = [0.3, 0.2, 0.1]
A = imwip.AffineWarpingOperator3D(
    im_shape=(im_size, im_size, im_size),
    rotation=true_rot,
    translation=true_trans,
    centered=True
).A
I = np.eye(3)
P = (I - A) @ np.linalg.inv(I + A)
true_cayley = [P[0, 1], P[0, 2], P[1, 2]]
shepp = tomopy.shepp3d(im_size).astype(np.float32)

# motion operator
M = lambda cayley, trans: imwip.AffineWarpingOperator3D(
    im_shape=(im_size, im_size, im_size),
    cayley=cayley,
    translation=trans,
    centered=True
)

# simulate
x = shepp.ravel()
b = M(true_cayley, true_trans) @ x

# solve
# ---------------------

# objective function
def f(cayley, trans):
    res = M(cayley, trans) @ x - b
    return 1/2 * np.dot(res, res)

# gradient of objective function
def grad_f(cayley, trans):
    res = M(cayley, trans) @ x - b
    print(np.linalg.norm(res))
    dMx_cayley, dMx_trans = imwip.diff(M(cayley, trans), x, to=["cayley", "trans"])
    grad_cayley = dMx_cayley.T @ res
    grad_trans = dMx_trans.T @ res
    return grad_cayley, grad_trans

# calback function: print motion parameters each iteration
callback = lambda cayley, trans: print(cayley, trans)

# initial guess: zero
cayley0 = [0.0, 0.0, 0.0]
trans0 = [25.0, -10, 15]
x0 = np.zeros(im_size**3, dtype=np.float32)

# solve
cayley, trans = imwip.split_barzilai_borwein(
    grad_f,
    x0=(cayley0, trans0),
    max_iter=30,
    verbose=True,
    callback=callback
)

print(cayley, trans)
print(true_cayley)