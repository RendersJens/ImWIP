# This example shows how to do simple 2D affine image warping

import numpy as np
from matplotlib import pyplot as plt
from skimage import transform
import imwip
import sys

# We use a sample image from tomopy, but you can replace this with any image
import tomopy
im_size = 128
shepp = tomopy.shepp3d(im_size).astype(np.float32)

# example affine transform
A = np.array([[np.cos(np.pi/6), -np.sin(np.pi/6), 0],
              [np.sin(np.pi/6),  np.cos(np.pi/6), 0],
              [0,                0.5,             1]], dtype=np.float32)
b = np.array([1, 2, 3], dtype=np.float32)

# an example DVF
u = 10*np.repeat(np.sin(np.linspace(0,4*np.pi, im_size, dtype=np.float32)), im_size**2).reshape((im_size,)*3)
v = 8*np.repeat(np.cos(np.linspace(0,4*np.pi, im_size, dtype=np.float32)), im_size**2).reshape((im_size,)*3)
w = 2*np.repeat(np.cos(np.linspace(0,4*np.pi, im_size, dtype=np.float32)), im_size**2).reshape((im_size,)*3)

M = imwip.warp_3D_matrix(shepp.shape, u, v, w, degree=1)

warped = (M @ shepp.ravel()).reshape(shepp.shape)
warped2 = imwip.warp_3D(shepp, u, v, w, degree=1)

# plots
plt.figure()
plt.title("original")
plt.imshow(shepp[:, im_size//2, :], cmap="gray")
plt.colorbar()

plt.figure()
plt.title("warped")
plt.imshow(warped[:, im_size//2, :], cmap="gray")
plt.colorbar()

plt.figure()
plt.title("warped2")
plt.imshow(warped2[:, im_size//2, :], cmap="gray")
plt.colorbar()


plt.show()