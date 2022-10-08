"""
:file:      operators_dvf.py
:brief:     DVF based warping operators. These operators provide
            high level acces to the warping algorithms of ImWIP.
:date:      20 DEC 2021
:author:    Jens Renders
            imec-Visionlab
            University of Antwerp
            jens.renders@uantwerpen.be
"""


import numpy as np
from scipy.sparse.linalg import LinearOperator
import pylops
from .gradient_operators import SquareGradientOperator
try:
    from imwip_cuda import (warp_2D,
                            warp_3D,
                            adjoint_warp_2D,
                            adjoint_warp_3D,
                            grad_warp_2D,
                            grad_warp_3D,
                            partial_grad_warp_3D)
except ModuleNotFoundError:
    pass#raise NotImplementedError("dvf operators currently only supported on linux")


class WarpingOperator2D(LinearOperator):

    def __init__(self, u, v, ui=None, vi=None, adjoint_type="exact", degree=3):
        self.dtype = np.dtype('float32')
        self.shape = (u.size, u.size)
        self.u = u
        self.v = v
        self.ui = ui
        self.vi = vi
        self.adjoint_type = adjoint_type
        self.degree = degree

    def _matvec(self, x):

        # we expect the input as flattened array, so we reshape it
        x = x.reshape(self.u.shape)

        # preform the warp
        x_warped = warp_2D(x, self.u, self.v, degree=self.degree)

        # return as flattened array
        return x_warped.ravel()

    def _rmatvec(self, x_warped):
        
        # we expect the input as flattened array, so we reshape it
        x_warped = x_warped.reshape(self.u.shape)

        # preform the adjoint warp
        if self.adjoint_type == "exact":
            x = adjoint_warp_2D(x_warped, self.u, self.v, degree=self.degree)
        elif self.adjoint_type == "negative":
            x = warp_2D(x_warped, -self.u, -self.v, degree=self.degree)
        elif self.adjoint_type == "inverse":
            if self.ui is None or self.vi is None:
                raise ValueError("adjoint type 'inverse' requires ui and vi to be given.")
            x = warp_2D(x_warped, self.ui, self.vi, degree=self.degree)
        else:
            raise NotImplementedError("adjoint type should be 'exact', 'inverse' or 'negative'")
        
        # return as flattened array
        return x.ravel()


class WarpingOperator3D(LinearOperator):

    def __init__(self, u, v, w, ui=None, vi=None, wi=None, adjoint_type="exact", degree=3):
        self.dtype = np.dtype('float32')
        self.shape = (u.size, u.size)
        self.u = u
        self.v = v
        self.w = w
        self.ui = ui
        self.vi = vi
        self.wi = wi
        self.adjoint_type = adjoint_type
        self.degree = degree

    def _matvec(self, x):

        # we expect the input as flattened array, so we reshape it
        x = x.reshape(self.u.shape)

        # preform the warp
        x_warped = warp_3D(x, self.u, self.v, self.w, degree=self.degree)

        # return as flattened array
        return x_warped.ravel()

    def _rmatvec(self, x_warped):
        
        # we expect the input as flattened array, so we reshape it
        x_warped = x_warped.reshape(self.u.shape)

        # preform the adjoint warp
        if self.adjoint_type == "exact":
            x = adjoint_warp_3D(x_warped, self.u, self.v, self.w, degree=self.degree)
        elif self.adjoint_type == "negative":
            x = warp_3D(x_warped, -self.u, -self.v, -self.w, degree=self.degree)
        elif self.adjoint_type == "inverse":
            if self.ui is None or self.vi is None:
                raise ValueError("adjoint type 'inverse' requires ui, vi and wi to be given.")
            x = warp_3D(x_warped, self.ui, self.vi, self.wi, degree=self.degree)
        else:
            raise NotImplementedError("adjoint type should be 'exact', 'inverse' or 'negative'")
        
        # return as flattened array
        return x.ravel()


def diff_warp_2D(f, u, v, approx=False):
    if approx:
        warped = warp_2D(f, u, v)
        gradx, grady = np.gradient(warped)
    else:
        gradx, grady = grad_warp_2D(f, u, v)
    diffx = pylops.Diagonal(gradx.ravel(), dtype=np.float32)
    diffy = pylops.Diagonal(grady.ravel(), dtype=np.float32)
    return pylops.VStack([diffx, diffy], dtype=np.float32)


def diff_warp_3D(f, u, v, w, approx=False):
    if approx:
        warped = warp_3D(f, u, v, w)
        grad = np.gradient(warped, axis=to)
    else:
        gradx, grady, gradz = grad_warp_3D(f, u, v, w)
    diffx = pylops.Diagonal(gradx.ravel(), dtype=np.float32)
    diffy = pylops.Diagonal(grady.ravel(), dtype=np.float32)
    diffz = pylops.Diagonal(gradz.ravel(), dtype=np.float32)
    return pylops.VStack([diffx, diffy, diffz], dtype=np.float32)


def partial_diff_warp_3D(f, u, v, w, to, approx=False):
    if approx:
        warped = warp_3D(f, u, v, w)
        grad = np.gradient(warped, axis=to)
    else:
        grad = partial_grad_warp_3D(f, u, v, w, to)
    diff = pylops.Diagonal(grad.ravel(), dtype=np.float32)
    return diff