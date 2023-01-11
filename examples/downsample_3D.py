import numpy as np
import tomopy
import pylops
from matplotlib import pyplot as plt
from imwip.numba_backend.downsample_algorithms import downsample_3D, adjoint_downsample_3D

im_size = 256
shepp = tomopy.shepp3d(im_size).astype(np.float32)

shepp_lr = np.zeros((100, 100, 100), dtype=np.float32)
downsample_3D(shepp, shepp_lr)

shepp_adjoint = np.zeros(shepp.shape, dtype=np.float32)
adjoint_downsample_3D(shepp_adjoint, shepp_lr)

plt.figure()
plt.imshow(shepp[shepp.shape[0]//2, :, :], cmap="gray")

plt.figure()
plt.imshow(shepp_lr[shepp_lr.shape[0]//2, :, :], cmap="gray")

plt.figure()
plt.imshow(shepp_adjoint[shepp.shape[0]//2, :, :], cmap="gray")

plt.show()