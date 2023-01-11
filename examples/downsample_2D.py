import numpy as np
import tomopy
import pylops
from matplotlib import pyplot as plt
from imwip.numba_backend.downsample_algorithms import downsample_2D, adjoint_downsample_2D

im_size = 256
shepp = tomopy.shepp2d(im_size)[0].astype(np.float32)

shepp_lr = np.zeros((100, 100), dtype=np.float32)
downsample_2D(shepp, shepp_lr)

shepp_adjoint = np.zeros(shepp.shape, dtype=np.float32)
adjoint_downsample_2D(shepp_adjoint, shepp_lr)

plt.figure()
plt.imshow(shepp, cmap="gray")

plt.figure()
plt.imshow(shepp_lr, cmap="gray")

plt.figure()
plt.imshow(shepp_adjoint, cmap="gray")

plt.show()