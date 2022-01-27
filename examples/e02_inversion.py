# This example shows how to do image reconstruction using the
# 2D image warping operator

import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse.linalg import lsqr
import imwip

# we start by preforming the same warp as in s01_basic_warping.py
import tomopy
im_size = 512
shepp = tomopy.shepp2d(im_size)[0].astype(np.float32)
u = 10*np.repeat(np.sin(np.linspace(0,4*np.pi, im_size, dtype=np.float32)), im_size).reshape((im_size, im_size))
v = 8*np.repeat(np.cos(np.linspace(0,4*np.pi, im_size, dtype=np.float32)), im_size).reshape((im_size, im_size))

# We can preform the warp with more high level syntax,
# as if warping is a matrix vector multiplication (which it is).
A = imwip.WarpingOperator2D(u, v)
b = A @ shepp.ravel() # @ is matrix mult
warped_shepp = b.reshape(shepp.shape)

# we will now try to reconstruct shepp form the warped version "warped_shepp"
# this reconstruction problem can be fromulated as
# Ax = b
# where A is the warping operator, and b is the warped shepp
x = lsqr(A, b, iter_lim=30)[0] # least squares solver of scipy
reconstruction = x.reshape(shepp.shape)

plt.figure()
plt.title("original")
plt.imshow(shepp, cmap="gray")
plt.colorbar()

plt.figure()
plt.title("warped")
plt.imshow(warped_shepp, cmap="gray")
plt.colorbar()

plt.figure()
plt.title("reconstructed")
plt.imshow(reconstruction, cmap="gray")
plt.colorbar()

plt.show()