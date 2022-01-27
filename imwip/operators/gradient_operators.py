"""
:file:      gradient_operators.py
:brief:     Operators for computing image gradients
:date:      20 DEC 2021
:author:    Jens Renders
            imec-Visionlab
            University of Antwerp
            jens.renders@uantwerpen.be
"""

import numpy as np
# from math import prod
from functools import reduce
prod = lambda *factors: reduce(lambda x, y: x*y, factors)
from scipy.sparse.linalg import LinearOperator

class GradientOperator(LinearOperator):
    def __init__(self, n, dtype=None):
        self.n = n
        self.shape = (n-1, n)
        if dtype is None:
            self.dtype = np.float32
        else:
            self.dtype = dtype

    def _matvec(self, x):
        return np.diff(x.ravel())

    def _rmatvec(self, x):
        x = x.ravel()
        y = np.zeros(self.n)
        y[1:-1] = -np.diff(x)
        y[0] = -x[0]
        y[-1] = x[-1]
        return y

class SquareGradientOperator(LinearOperator):
    def __init__(self, n, dtype=None):
        self.n = n
        self.shape = (n, n)
        if dtype is None:
            self.dtype = np.float32
        else:
            self.dtype = dtype

    def _matvec(self, x):
        x = x.ravel()
        y = np.zeros(self.n)
        y[1:] = np.diff(x)
        y[0] = x[0]
        return y

    def _rmatvec(self, x):
        x = x.ravel()
        y = np.zeros(self.n)
        y[:-1] = -np.diff(x)
        y[-1] = x[-1]
        return y


class GradientOperator3DX(LinearOperator):
    def __init__(self, im_shape, dtype=None):
        self.im_shape = im_shape
        self.shape = ((im_shape[0]-1)*im_shape[1]*im_shape[2], prod(*im_shape))
        if dtype is None:
            self.dtype = np.float32
        else:
            self.dtype = dtype

    def _matvec(self, x):
        return np.diff(x.reshape(self.im_shape), axis=0)

    def _rmatvec(self, x):
        x = x.reshape((self.im_shape[0]-1, *self.im_shape[1:]))
        y = np.zeros(self.im_shape, dtype=self.dtype)
        y[1:-1, :, :] = -np.diff(x, axis=0)
        y[0, :, :] = -x[0, :, :]
        y[-1, :, :] = x[-1, :, :]
        return y


class GradientOperator3DY(LinearOperator):
    def __init__(self, im_shape, dtype=None):
        self.im_shape = im_shape
        self.shape = (im_shape[0]*(im_shape[1]-1)*im_shape[2], prod(*im_shape))
        if dtype is None:
            self.dtype = np.float32
        else:
            self.dtype = dtype

    def _matvec(self, x):
        return np.diff(x.reshape(self.im_shape), axis=1)

    def _rmatvec(self, x):
        x = x.reshape((self.im_shape[0], self.im_shape[1]-1, self.im_shape[2]))
        y = np.zeros(self.im_shape, dtype=self.dtype)
        y[:, 1:-1, :] = -np.diff(x, axis=1)
        y[:, 0, :] = -x[:, 0, :]
        y[:, -1, :] = x[:, -1, :]
        return y


class GradientOperator3DZ(LinearOperator):
    def __init__(self, im_shape, dtype=None):
        self.im_shape = im_shape
        self.shape = (im_shape[0]*im_shape[1]*(im_shape[2]-1), prod(*im_shape))
        if dtype is None:
            self.dtype = np.float32
        else:
            self.dtype = dtype

    def _matvec(self, x):
        return np.diff(x.reshape(self.im_shape), axis=2)

    def _rmatvec(self, x):
        x = x.reshape((*self.im_shape[:-1], self.im_shape[2]-1))
        y = np.zeros(self.im_shape, dtype=self.dtype)
        y[:, :, 1:-1] = -np.diff(x, axis=2)
        y[:, :, 0] = -x[:, :, 0]
        y[:, :, -1] = x[:, :, -1]
        return y