# This example shows how to use the gpu warping functions

import numpy as np
from matplotlib import pyplot as plt
from time import time
import imwip

# We use a sample image from tomopy, but you can replace this with any image
import tomopy
im_size = 512
shepp = tomopy.shepp3d(im_size).astype(np.float32)

# example affine transform
A = np.array([[np.cos(np.pi/6), -np.sin(np.pi/6), 0],
              [np.sin(np.pi/6),  np.cos(np.pi/6), 0],
              [0,                0.5,             1]], dtype=np.float32)
b = np.array([1, 2, 3], dtype=np.float32)

# linear backward warp
linear_warped_shepp = imwip.affine_warp(shepp, A, b, degree=1)

# cubic backward warp
t0 = time()
cubic_warped_shepp = imwip.affine_warp(shepp, A, b)
print(time()-t0)

# plots
plt.figure()
plt.title("original")
plt.imshow(shepp[im_size//2, :, :], cmap="gray")
plt.colorbar()

plt.figure()
plt.title("linear warped")
plt.imshow(linear_warped_shepp[im_size//2, :, :], cmap="gray")
plt.colorbar()

plt.figure()
plt.title("cubic warped")
plt.imshow(cubic_warped_shepp[im_size//2, :, :], cmap="gray")
plt.colorbar()

plt.show()