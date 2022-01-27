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

print(A)
print(b)

# linear backward warp
linear_warped_shepp = imwip.affine_warp_2D(shepp, A, b, degree=1)

# cubic backward warp
cubic_warped_shepp = imwip.affine_warp_2D(shepp, A, b)

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