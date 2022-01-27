# This example shows a more advanced application: motion estimation and correction in 4D-CT

import numpy as np
import astra
import tomopy
import pylops
from matplotlib import pyplot as plt
import imwip

im_size = 512
translation = [50, -20]
rotation = 0.3

# create operators
#--------------------------

# CT system (astra): two interleaving subscans
vol_geom = astra.create_vol_geom(im_size, im_size)
angles_1 = np.linspace(0, np.pi, 100)[0::2]
angles_2 = np.linspace(0, np.pi, 100)[1::2]
proj_geom_1 = astra.create_proj_geom('parallel', 1, im_size, angles_1)
proj_id_1 = astra.create_projector('cuda', proj_geom_1, vol_geom)
W_1 = astra.optomo.OpTomo(proj_id_1)
proj_geom_2 = astra.create_proj_geom('parallel', 1, im_size, angles_2)
proj_id_2 = astra.create_projector('cuda', proj_geom_2, vol_geom)
W_2 = astra.optomo.OpTomo(proj_id_2)
W = pylops.BlockDiag([W_1, W_2])

# motion operators
M_1 = pylops.Identity(im_size**2, dtype=np.dtype('float32'))
M_2 = lambda rot, trans: imwip.AffineWarpingOperator2D(
    im_shape=(im_size, im_size),
    rotation=rot,
    translation=trans,
    adjoint_type="inverse")
M = lambda rot, trans: pylops.VStack([M_1, M_2(rot, trans)])

# create simulated data
# ----------------------

#phantoms
shepp = tomopy.shepp2d(im_size)[0].astype(np.float32)/255
moved_shepp =  np.clip(M_2(rotation, translation) @ shepp.ravel(), 0, 1)

# project phantoms according to the 2 subscans
p_1 = W_1 @ shepp.ravel()
p_2 = W_2 @ moved_shepp
p = np.concatenate([p_1, p_2])

# solve
# ---------------------

# objective function
def f(x, rot, trans):
    rot = rot[0]
    res = W @ (M(rot, trans) @ x) - p
    return 1/2 * np.dot(res, res)

# gradient of objective function
def grad_f(x, rot, trans):
    rot = rot[0]

    WTres = W.T @ (W @ (M(rot, trans) @ x) - p)

    dMx_rot, dMx_trans = imwip.diff(M(rot, trans), x, to=["rot", "trans"])
    grad_rot = dMx_rot.T @ WTres
    grad_trans = dMx_trans.T @ WTres
    grad_x = M(rot, trans).T @ WTres

    return grad_x, grad_rot, grad_trans

# calback function: print motion parameters each iteration
callback = lambda x, rot, trans: print(*rot, trans)

# initial guess: zero for the motion parameters,
#                and a scaled backprojection for the image
rot0 = [0.0]
trans0 = [0.0, 0.0]
x0 = np.zeros(im_size**2, dtype=np.float32)

# solve
x, rot, trans = imwip.split_barzilai_borwein(
    grad_f,
    x0=(x0, rot0, trans0),
    bounds=((0, 1), None, None),
    max_iter=150,
    verbose=False,
    callback=callback)

rec = x.reshape((im_size, im_size))
print(rot, trans)


# reconstruction without motion correction for comparison
# --------------------------------------------------------

W = pylops.VStack([W_1, W_2])

def grad_f(x):
    res = W @ x - p
    grad_x = W.adjoint() @ res
    return grad_x

x0 = W.T @ p
x0 /= x0.max()
x = imwip.barzilai_borwein(grad_f, x0=x0, bounds=(0,1), max_iter=100)
bad_rec = x.reshape((im_size, im_size))


# plots
plt.figure()
plt.title("original in first subscan")
plt.imshow(shepp, cmap="gray")
plt.colorbar()

plt.figure()
plt.title("original in second subscan")
plt.imshow(moved_shepp.reshape(shepp.shape), cmap="gray")
plt.colorbar()

plt.figure()
plt.title("motion corrected reconstruction")
plt.imshow(rec, cmap="gray")
plt.colorbar()

plt.figure()
plt.title("motion corrected reconstruction with estimated motion applied")
plt.imshow(np.clip(M_2(*rot, trans) @ rec.ravel(), 0, 255).reshape(rec.shape), cmap="gray")
plt.colorbar()

plt.figure()
plt.title("ordinary reconstruction")
plt.imshow(bad_rec, cmap="gray")
plt.colorbar()

plt.show()