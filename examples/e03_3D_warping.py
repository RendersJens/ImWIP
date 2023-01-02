# This example shows how to do simple 3D image warping along a vectorfield

import numpy as np
from matplotlib import pyplot as plt
import imwip

# We use a sample image from tomopy, but you can replace this with any image
import tomopy
im_size = 256
shepp = tomopy.shepp3d(im_size).astype(np.float32)

# an example DVF
u = 10*np.repeat(np.sin(np.linspace(0,4*np.pi, im_size, dtype=np.float32)), im_size**2).reshape((im_size,)*3)
v = 8*np.repeat(np.cos(np.linspace(0,4*np.pi, im_size, dtype=np.float32)), im_size**2).reshape((im_size,)*3)
w = 2*np.repeat(np.cos(np.linspace(0,4*np.pi, im_size, dtype=np.float32)), im_size**2).reshape((im_size,)*3)

# linear backward warp
linear_warped_shepp = imwip.warp(shepp, u, v, w)

# cubic backward warp
cubic_warped_shepp = imwip.warp(shepp, u, v, w, degree=3)

# a cubic warp can produce values outside of the original range
np.clip(cubic_warped_shepp, 0, 1, out=cubic_warped_shepp)

# plots
plt.figure()
plt.title("original")
plt.imshow(shepp[:, im_size//2, :], cmap="gray")
plt.colorbar()

plt.figure()
plt.title("linear warped")
plt.imshow(linear_warped_shepp[:, im_size//2, :], cmap="gray")
plt.colorbar()

plt.figure()
plt.title("cubic warped")
plt.imshow(cubic_warped_shepp[:, im_size//2, :], cmap="gray")
plt.colorbar()

plt.show()