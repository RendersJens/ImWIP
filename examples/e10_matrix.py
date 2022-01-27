# This example shows how to do simple 2D affine image warping

import numpy as np
from matplotlib import pyplot as plt
from skimage import transform
import imwip
import sys

# We use a sample image from tomopy, but you can replace this with any image
import tomopy
im_size = 512
shepp = tomopy.shepp2d(im_size)[0].astype(np.float32)

# an example affine transform using skimage utility function
tform = transform.AffineTransform(rotation=0.1).params.astype(np.float32)
A = tform[:2,:2].copy()
b = tform[2, :2]

# an example DVF
u = 10*np.repeat(np.sin(np.linspace(0,4*np.pi, im_size, dtype=np.float32)), im_size).reshape((im_size, im_size))
v = 8*np.repeat(np.cos(np.linspace(0,4*np.pi, im_size, dtype=np.float32)), im_size).reshape((im_size, im_size))

M = imwip.warp_2D_matrix(shepp.shape, u, v, degree=1)

warped = (M @ shepp.ravel()).reshape(shepp.shape)
warped2 = imwip.warp_2D(shepp, u, v, degree=1)

# plots
plt.figure()
plt.title("original")
plt.imshow(shepp, cmap="gray")
plt.colorbar()

plt.figure()
plt.title("warped")
plt.imshow(warped, cmap="gray")
plt.colorbar()

plt.figure()
plt.title("warped2")
plt.imshow(warped2, cmap="gray")
plt.colorbar()


plt.show()