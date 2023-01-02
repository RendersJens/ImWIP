"""
:file:      operators_dvf.py
:brief:     High level access to the DVF warping functions through linear operators.
:author:    Jens Renders
"""


import numpy as np
import imwip
from scipy.sparse.linalg import LinearOperator
import pylops
from .gradient_operators import SquareGradientOperator


class WarpingOperator2D(LinearOperator):
    """
    A 2D warping operator represented as a scipy LinearOperator.

    :param u: First component of the DVF describing the warp
    :param v: Second component of the DVF describing the warp
    :param degree: Degree of the splines used for interpolation
    :param adjoint_type: Method to compute the adjoint. Defaults to ``exact``.
    :param backend: Whether to use the cpp or numba backend. If None, ``cpp`` will be used
        if available, else "numba"

    :type u: :class:`numpy.ndarray`
    :type v: :class:`numpy.ndarray`
    :type degree: 1 or 3, optional
    :type adjoint_type: ``exact``, ``negative`` or ``inverse``
    :type backend: ``cpp`` or ``numba``, optional
    """

    def __init__(
            self,
            u,
            v,
            ui=None,
            vi=None,
            degree=3,
            adjoint_type="exact",
            backend=None
        ):
        self.dtype = np.dtype('float32')
        self.shape = (u.size, u.size)
        self.u = u
        self.v = v
        self.ui = ui
        self.vi = vi
        self.degree = degree
        self.adjoint_type = adjoint_type
        self.backend = backend

    def _matvec(self, x):

        # we expect the input as flattened array, so we reshape it
        x = x.reshape(self.u.shape)

        # perform the warp
        x_warped = imwip.warp(
            x,
            self.u,
            self.v,
            degree=self.degree,
            backend=self.backend
        )

        # return as flattened array
        return x_warped.ravel()

    def _rmatvec(self, x_warped):
        
        # we expect the input as flattened array, so we reshape it
        x_warped = x_warped.reshape(self.u.shape)

        # perform the adjoint warp
        if self.adjoint_type == "exact":
            x = imwip.adjoint_warp(
                x_warped,
                self.u,
                self.v,
                degree=self.degree,
                backend=self.backend
            )
        elif self.adjoint_type == "negative":
            x = imwip.warp(
                x_warped,
                -self.u,
                -self.v,
                degree=self.degree,
                backend=self.backend
            )
        elif self.adjoint_type == "inverse":
            if self.ui is None or self.vi is None:
                raise ValueError("adjoint type 'inverse' requires ui and vi to be given.")
            x = imwip.warp(
                x_warped,
                self.ui,
                self.vi,
                degree=self.degree,
                backend=self.backend
            )
        else:
            raise NotImplementedError("adjoint type should be 'exact', 'inverse' or 'negative'")
        
        # return as flattened array
        return x.ravel()


class WarpingOperator3D(LinearOperator):

    def __init__(
            self, 
            u,
            v,
            w,
            ui=None,
            vi=None,
            wi=None,
            degree=3,
            adjoint_type="exact",
            backend=None
        ):
        self.dtype = np.dtype('float32')
        self.shape = (u.size, u.size)
        self.u = u
        self.v = v
        self.w = w
        self.ui = ui
        self.vi = vi
        self.wi = wi
        self.degree = degree
        self.adjoint_type = adjoint_type
        self.backend = backend

    def _matvec(self, x):

        # we expect the input as flattened array, so we reshape it
        x = x.reshape(self.u.shape)

        # perform the warp
        x_warped = imwip.warp(
            x,
            self.u,
            self.v,
            self.w,
            degree=self.degree,
            backend=self.backend
        )

        # return as flattened array
        return x_warped.ravel()

    def _rmatvec(self, x_warped):
        
        # we expect the input as flattened array, so we reshape it
        x_warped = x_warped.reshape(self.u.shape)

        # perform the adjoint warp
        if self.adjoint_type == "exact":
            x = imwip.adjoint_warp(
                x_warped,
                self.u,
                self.v,
                self.w,
                degree=self.degree,
                backend=self.backend
            )
        elif self.adjoint_type == "negative":
            x = imwip.warp(
                x_warped,
                -self.u,
                -self.v,
                -self.w,
                degree=self.degree,
                backend=self.backend
            )
        elif self.adjoint_type == "inverse":
            if self.ui is None or self.vi is None or wi is None:
                raise ValueError("adjoint type 'inverse' requires ui, vi and wi to be given.")
            x = imwip.warp(
                x_warped,
                self.ui,
                self.vi,
                self.wi,
                degree=self.degree,
                backend=self.backend
            )
        else:
            raise NotImplementedError("adjoint type should be 'exact', 'inverse' or 'negative'")
        
        # return as flattened array
        return x.ravel()


def diff_warping_operator_2D(f, u, v, approx=False, backend=None):
    if approx:
        warped = imwip.warp(f, u, v, backend=backend)
        gradx, grady = np.gradient(warped)
    else:
        gradx, grady = imwip.diff_warp(f, u, v, backend=backend)
    diffx = pylops.Diagonal(gradx.ravel(), dtype=np.float32)
    diffy = pylops.Diagonal(grady.ravel(), dtype=np.float32)
    return pylops.VStack([diffx, diffy], dtype=np.float32)


def diff_warping_operator_3D(f, u, v, w, approx=False, backend=None):
    if approx:
        warped = imwip.warp(f, u, v, w, backend=backend)
        gradx, grady, gradz = np.gradient(warped)
    else:
        gradx, grady, gradz = imwip.diff_warp(f, u, v, w, backend=backend)
    diffx = pylops.Diagonal(gradx.ravel(), dtype=np.float32)
    diffy = pylops.Diagonal(grady.ravel(), dtype=np.float32)
    diffz = pylops.Diagonal(gradz.ravel(), dtype=np.float32)
    return pylops.VStack([diffx, diffy, diffz], dtype=np.float32)


def partial_diff_warping_operator_3D(f, u, v, w, to, approx=False, backend=None):
    if approx:
        warped = imwip.warp(f, u, v, w, backend=backend)
        grad = np.gradient(warped, axis=to)
    else:
        grad = imwip.partial_diff_warp(f, u, v, w, to, backend=backend)
    diff = pylops.Diagonal(grad.ravel(), dtype=np.float32)
    return diff