import numpy as np
import astra
import tomopy
import pylops
from matplotlib import pyplot as plt
import imwip
from scipy.ndimage import center_of_mass

im_size = 256
true_trans = [25, -10, 15]
true_rot = [0.3, 0.2, 0.1]
A = imwip.AffineWarpingOperator3D(
    im_shape=(im_size, im_size, im_size),
    rotation=true_rot,
    translation=true_trans,
    centered=True
).A
shepp = tomopy.shepp3d(im_size).astype(np.float32)

# motion operator for simulation
M = imwip.AffineWarpingOperator3D(
    im_shape=(im_size, im_size, im_size),
    rotation=true_rot,
    translation=true_trans,
    centered=True
)

# simulate
b = M @ shepp.ravel()

# solve
# ---------------------

com = np.array(center_of_mass(b.reshape(shepp.shape)), dtype=np.float32)
trans = np.array(center_of_mass(shepp)) - com
T = imwip.AffineWarpingOperator3D(
    im_shape=(im_size, im_size, im_size),
    translation=trans,
    centered=True
)
x = T @ shepp.ravel()

R = lambda rot: imwip.AffineWarpingOperator3D(
    im_shape=(im_size, im_size, im_size),
    rotation=rot,
    center=com,
    centered=True
)

# objective function
def f(rot):
    res = R(rot) @ x - b
    return 1/2 * np.dot(res, res)

# gradient of objective function
def grad_f(rot):
    res = R(rot) @ x - b
    print(np.linalg.norm(res))
    dMx_rot = imwip.diff(R(rot), x, to="rot")
    grad_rot = dMx_rot.T @ res
    return grad_rot

# calback function: print motion parameters each iteration
callback = lambda rot: print(rot, R(rot).A @ trans)

# initial guess: zero
rot0 = [0.0, 0.0, 0.0]

# solve
rot = imwip.barzilai_borwein(
    grad_f,
    x0=rot0,
    max_iter=20,
    verbose=True,
    callback=callback
)

print(rot)
print(true_rot)