# This example shows how to do 3D image reconstruction using the
# 3D warping operator

import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse.linalg import lsqr
import imwip


# we start by preforming the same warp as in s04_3D_warping.py
import tomopy
im_size = 256
shepp = tomopy.shepp3d(im_size).astype(np.float32)
u = 10*np.repeat(np.sin(np.linspace(0,4*np.pi, im_size, dtype=np.float32)), im_size**2).reshape((im_size,)*3)
v = 8*np.repeat(np.cos(np.linspace(0,4*np.pi, im_size, dtype=np.float32)), im_size**2).reshape((im_size,)*3)
w = 2*np.repeat(np.cos(np.linspace(0,4*np.pi, im_size, dtype=np.float32)), im_size**2).reshape((im_size,)*3)

# We can preform the warp with more high level syntax,
# as if warping is a matrix vector multiplication (which it is).
A = imwip.WarpingOperator3D(u, v, w)
b = A @ shepp.ravel() # @ is matrix mult
warped_shepp = b.reshape(shepp.shape)

# we will now try to reconstruct shepp form the warped version "warped_shepp"
# this reconstruction problem can be fromulated as
# Ax = b
# where A is the warping operator, and b is the warped shepp
x = lsqr(A, b, iter_lim=30, show=True)[0] # least squares solver of scipy
reconstruction = x.reshape(shepp.shape)

plt.figure()
plt.title("original")
plt.imshow(shepp[:, im_size//2, :], cmap="gray")
plt.colorbar()

plt.figure()
plt.title("warped")
plt.imshow(warped_shepp[:, im_size//2, :], cmap="gray")
plt.colorbar()

plt.figure()
plt.title("reconstructed")
plt.imshow(reconstruction[:, im_size//2, :], cmap="gray")
plt.colorbar()

plt.show()