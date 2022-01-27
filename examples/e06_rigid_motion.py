# This example shows how to do simple 2D affine image warping

import numpy as np
from matplotlib import pyplot as plt
import imwip

# We use a sample image from tomopy, but you can replace this with any image
import tomopy
im_size = 512
shepp = tomopy.shepp2d(im_size)[0].astype(np.float32)

# an example rigid motion operator: rotation = 0.2 rad, translation = (0, 0)
# by default, the center of rotation is the center of the image
# by default the warp is cubic
M = imwip.AffineWarpingOperator2D((im_size, im_size), rotation=0.2, translation=(0, 0))

warped_shepp = (M @ shepp.ravel()).reshape(shepp.shape) # @ is matrix mult

# a cubic warp can produce values outside of the original range
np.clip(warped_shepp, 0, 255, out=warped_shepp)

# plots
plt.figure()
plt.title("original")
plt.imshow(shepp, cmap="gray")
plt.colorbar()

plt.figure()
plt.title("warped")
plt.imshow(warped_shepp, cmap="gray")
plt.colorbar()

plt.show()