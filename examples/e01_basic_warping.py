# This example shows how to do simple 2D image warping along a vectorfield

import numpy as np
from matplotlib import pyplot as plt
import imwip

# We use a sample image from tomopy, but you can replace this with any image
import tomopy
im_size = 512
shepp = tomopy.shepp2d(im_size)[0].astype(np.float32)

# an example DVF
u = 10*np.repeat(np.sin(np.linspace(0,4*np.pi, im_size, dtype=np.float32)), im_size).reshape((im_size, im_size))
v = 8*np.repeat(np.cos(np.linspace(0,4*np.pi, im_size, dtype=np.float32)), im_size).reshape((im_size, im_size))

# linear backward warp
linear_warped_shepp = imwip.warp(shepp, u, v, degree=1)

# cubic backward warp
cubic_warped_shepp = imwip.warp(shepp, u, v, degree=3)

# a cubic warp can produce values outside of the original range
np.clip(cubic_warped_shepp, 0, 255, out=cubic_warped_shepp)

# plots
plt.figure()
plt.title("original")
plt.imshow(shepp, cmap="gray")
plt.colorbar()

plt.figure()
plt.title("linear warped")
plt.imshow(linear_warped_shepp, cmap="gray")
plt.colorbar()

plt.figure()
plt.title("cubic warped")
plt.imshow(cubic_warped_shepp, cmap="gray")
plt.colorbar()

plt.show()