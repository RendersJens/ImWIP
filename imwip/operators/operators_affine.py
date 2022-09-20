"""
:file:      operators_affine.py
:brief:     Affine warping operators. These operators provide
            high level acces to the warping algorithms of ImWIP.
:date:      20 DEC 2021
:author:    Jens Renders
            imec-Visionlab
            University of Antwerp
            jens.renders@uantwerpen.be
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator
from collections.abc import Iterable
try:
    from imwip_cuda import (affine_warp_2D,
                            affine_warp_3D,
                            adjoint_affine_warp_2D,
                            adjoint_affine_warp_3D,
                            grad_affine_warp_2D,
                            grad_affine_warp_3D)
except ModuleNotFoundError:
    from imwip.numba import (affine_warp_2D,
                             affine_warp_3D,
                             adjoint_affine_warp_2D,
                             adjoint_affine_warp_3D,
                             grad_affine_warp_2D,
                             grad_affine_warp_3D)


class AffineWarpingOperator2D(LinearOperator):

    def __init__(
            self, im_shape,
            A=None,
            b=None,
            scale=None,
            rotation=None,
            translation=None,
            centered=True,
            adjoint_type="exact",
            derivative_type="exact",
            degree=3,
            indexing="ij"
        ):
        self.im_shape = im_shape
        self.im_size = im_shape[0]*im_shape[1]

        # accumulate the given transformations one by one
        self.dtype = np.dtype('float32')
        if A is not None:
            self.A = A.astype(np.float32)
        else:
            self.A = np.eye(2, dtype=np.float32)
        if b is not None:
            self.b = b.astype(np.float32)
        else:
            self.b = np.zeros(2, dtype=np.float32)
        if scale is not None:
            if not isinstance(scale, Iterable):
                scale = (scale, scale)
            self.scale = scale
            self.A = np.array([[scale[0],        0],
                               [0,        scale[1]]]) @ self.A
        if rotation is not None:
            self.rotation = rotation
            self.A = np.array([[ np.cos(rotation), np.sin(rotation)],
                               [-np.sin(rotation), np.cos(rotation)]]) @ self.A
        self.A = self.A.astype(np.float32)
        if translation is not None:
            self.translation = translation
            self.b += np.array(translation)

        self.centered = centered
        if self.centered:
            center = np.array(im_shape, dtype=np.float32)/2
            if indexing == "xy":
                center = np.flip(center)
            self.b = center - self.A @ center + self.b

        self.shape = (self.im_size, self.im_size)
        self.adjoint_type = adjoint_type
        self.derivative_type = derivative_type
        self.degree = degree
        if indexing not in ["ij", "xy"]:
            raise ValueError('indexing should be "xy" or "ij"')
        self.indexing = indexing

    def _matvec(self, x):

        # we expect the input as flattened array, so we reshape it
        x = x.reshape(self.im_shape)

        # preform the warp
        x_warped = affine_warp_2D(
            x,
            self.A,
            self.b,
            degree=self.degree,
            indexing=self.indexing
        )

        # return as flattened array
        return x_warped.ravel()

    def _rmatvec(self, x_warped):
        
        # we expect the input as flattened array, so we reshape it
        x_warped = x_warped.reshape(self.im_shape)

        # preform the adjoint warp
        if self.adjoint_type == "exact":
            x = adjoint_affine_warp_2D(
                x_warped,
                self.A,
                self.b,
                degree=self.degree,
                indexing=self.indexing
            )
        elif self.adjoint_type == "inverse":
            Ai = np.linalg.inv(self.A)
            x = affine_warp_2D(
                x_warped,
                Ai,
                -Ai @ self.b,
                degree=self.degree,
                indexing=self.indexing)
        else:
            raise NotImplementedError("adjoint type should be 'exact' or 'inverse'")
               
        # return as flattened array
        return x.ravel()

    def _derivative_scale(self, x, grad_x, grad_y, co_x, co_y):
        dxds = co_x.astype(np.float32)
        dyds = co_y.astype(np.float32)
        if self.centered:
            dxds -= self.im_shape[0]/2
            dyds -= self.im_shape[1]/2
        
        grad_sx = grad_x * dxds
        grad_sy = grad_y * dyds
        return np.vstack([grad_sx.ravel(), grad_sy.ravel()]).T

    def _derivative_zoom(self, x, grad_x, grad_y, co_x, co_y):
        grad_sx, grad_sy = self._derivative_scale(x, grad_x, grad_y, co_x, co_y).T
        grad_s = grad_sx + grad_sy
        return grad_s.reshape((-1, 1))

    def _derivative_rotation(self, x, grad_x, grad_y, co_x, co_y):
        dxdr = -co_x*np.sin(self.rotation) + co_y*np.cos(self.rotation)
        dydr = -co_x*np.cos(self.rotation) - co_y*np.sin(self.rotation)
        if self.centered:
            dxdr += self.im_shape[0]/2*np.sin(self.rotation) - self.im_shape[1]/2*np.cos(self.rotation)
            dydr += self.im_shape[0]/2*np.cos(self.rotation) + self.im_shape[1]/2*np.sin(self.rotation)

        grad_rotation = grad_x*dxdr + grad_y*dydr
        return grad_rotation.reshape((-1, 1))

    def _derivative_translation(self, x, grad_x, grad_y):
        return np.vstack([grad_x.ravel(), grad_y.ravel()]).T

    def derivative(self, x, to=["b"]):

        # we expect the input as flattened array, so we reshape it
        x = x.reshape(self.im_shape)
        if self.derivative_type == "exact":
            grad_x, grad_y = grad_affine_warp_2D(
                x,
                self.A,
                self.b,
                indexing=self.indexing
            )
        else:
            x_warped = affine_warp_2D(
                x,
                self.A,
                self.b,
                indexing=self.indexing
            )
            grad_x, grad_y = np.gradient(x_warped)

        if "rotation" in to or "rot" in to or "scale" in to or "zoom" in to:
            co_x, co_y = np.meshgrid(
                np.arange(self.im_shape[0]),
                np.arange(self.im_shape[1]),
                indexing=self.indexing
            )
        derivatives = []
        for var in to:
            if var == "scale":
                derivatives.append(self._derivative_scale(x, grad_x, grad_y, co_x, co_y))
            elif var == "zoom":
                derivatives.append(self._derivative_zoom(x, grad_x, grad_y, co_x, co_y))
            elif var in ["rot", "rotation"]:
                derivatives.append(self._derivative_rotation(x, grad_x, grad_y, co_x, co_y))
            elif var in ["trans", "translation", "b"]:
                derivatives.append(self._derivative_translation(x, grad_x, grad_y))
            else:
                derivatives.append(np.zeros((x.size, 0)))
        return derivatives


class AffineWarpingOperator3D(LinearOperator):

    def __init__(
            self, im_shape,
            A=None, 
            b=None,
            scale=None,
            rotation=None,
            translation=None,
            centered=True,
            center=None,
            adjoint_type="exact",
            degree=3,
            indexing="ij"
        ):
        self.im_shape = im_shape
        self.im_size = im_shape[0]*im_shape[1]*im_shape[2]

        # accumulate the given transformations one by one
        self.dtype = np.dtype('float32')
        if A is not None:
            self.A = A.astype(np.float32)
        else:
            self.A = np.eye(3, dtype=np.float32)
        if b is not None:
            self.b = b.astype(np.float32)
        else:
            self.b = np.zeros(3, dtype=np.float32)
        if scale is not None:
            if not isinstance(scale, Iterable):
                scale = (scale, scale, scale)
            self.scale = scale
            self.A = np.array([[scale[0],        0,        0],
                               [       0, scale[1],        0],
                               [       0,        0, scale[2]]]) @ self.A
        if rotation is not None:
            self.rotation = rotation

            rot_i = np.array([[1,                    0,                   0],
                              [0,  np.cos(rotation[0]), np.sin(rotation[0])],
                              [0, -np.sin(rotation[0]), np.cos(rotation[0])]])

            rot_j = np.array([[ np.cos(rotation[1]), 0, np.sin(rotation[1])],
                              [                   0, 1,                   0],
                              [-np.sin(rotation[1]), 0, np.cos(rotation[1])]])

            rot_k = np.array([[ np.cos(rotation[2]), np.sin(rotation[2]), 0],
                              [-np.sin(rotation[2]), np.cos(rotation[2]), 0],
                              [                   0,                   0, 1]])

            self.A = rot_k @ (rot_j @ (rot_i @ self.A))
        self.A = self.A.astype(np.float32)

        if translation is not None:
            self.translation = translation
            self.b += np.array(translation)

        self.centered = centered
        if center is None:
            self.center = np.array(im_shape, dtype=np.float32)/2
        else:
            self.center = center

        if self.centered:
            center = self.center
            if indexing == "xy":
                center = np.flip(center)
            self.b = center - self.A @ center + self.b

        self.shape = (self.im_size, self.im_size)
        self.adjoint_type = adjoint_type
        self.degree = degree
        if indexing not in ["ij", "xy"]:
            raise ValueError('indexing should be "xy" or "ij"')
        self.indexing = indexing

    def _matvec(self, x):

        # we expect the input as flattened array, so we reshape it
        x = x.reshape(self.im_shape)

        # preform the warp
        x_warped = affine_warp_3D(
            x,
            self.A,
            self.b,
            degree=self.degree,
            indexing=self.indexing
        )

        # return as flattened array
        return x_warped.ravel()

    def _rmatvec(self, x_warped):
        
        # we expect the input as flattened array, so we reshape it
        x_warped = x_warped.reshape(self.im_shape)

        # preform the adjoint warp
        x = adjoint_affine_warp_3D(
            x_warped,
            self.A,
            self.b,
            degree=self.degree,
            indexing=self.indexing
        )
        
        # return as flattened array
        return x.ravel()

    def _derivative_scale(
            self,
            x,
            grad_x,
            grad_y,
            grad_z,
            co_x,
            co_y,
            co_z
        ):
        dxds = co_x.astype(np.float32)
        dyds = co_y.astype(np.float32)
        dzds = co_z.astype(np.float32)
        if self.centered:
            dxds -= self.center[0]
            dyds -= self.center[1]
            dzds -= self.center[2]
        
        grad_sx = grad_x * dxds
        grad_sy = grad_y * dyds
        grad_sz = grad_z * dzds
        return np.vstack([grad_sx.ravel(), grad_sy.ravel(), grad_sz.ravel()]).T

    def _derivative_zoom(
            self,
            x,
            grad_x,
            grad_y,
            grad_z,
            co_x,
            co_y,
            co_z
        ):
        grad_sx, grad_sy, grad_sz = self._derivative_scale(x, grad_x, grad_y, co_x, co_y).T
        grad_s = grad_sx + grad_sy + grad_sz
        return grad_s.reshape((-1, 1))

    def _derivative_rotation(
            self,
            x,
            grad_x,
            grad_y,
            grad_z,
            co_x,
            co_y,
            co_z
        ):
        rotation = self.rotation
        rot_i = np.array([[1,                    0,                   0],
                          [0,  np.cos(rotation[0]), np.sin(rotation[0])],
                          [0, -np.sin(rotation[0]), np.cos(rotation[0])]])

        rot_j = np.array([[ np.cos(rotation[1]), 0, np.sin(rotation[1])],
                          [                   0, 1,                   0],
                          [-np.sin(rotation[1]), 0, np.cos(rotation[1])]])

        rot_k = np.array([[ np.cos(rotation[2]), np.sin(rotation[2]), 0],
                          [-np.sin(rotation[2]), np.cos(rotation[2]), 0],
                          [                   0,                   0, 1]])

        d_rot_i = np.array([[0,                    0,                    0],
                            [0, -np.sin(rotation[0]),  np.cos(rotation[0])],
                            [0, -np.cos(rotation[0]), -np.sin(rotation[0])]])

        d_rot_j = np.array([[-np.sin(rotation[1]), 0,  np.cos(rotation[1])],
                            [                   0, 0,                    0],
                            [-np.cos(rotation[1]), 0, -np.sin(rotation[1])]])

        d_rot_k = np.array([[-np.sin(rotation[2]), np.cos(rotation[2]), 0],
                            [-np.cos(rotation[2]),-np.sin(rotation[2]), 0],
                            [                   0,                   0, 0]])

        co = np.vstack([co_x.ravel(), co_y.ravel(), co_z.ravel()])

        dri = (rot_k @ (rot_j @ (d_rot_i @ co)))
        drj = (rot_k @ (d_rot_j @ (rot_i @ co)))
        drk = (d_rot_k @ (rot_j @ (rot_i @ co)))

        if self.centered:
            center = self.center.reshape((3, 1))
            dri -= rot_k @ (rot_j @ (d_rot_i @ center))
            drj -= rot_k @ (d_rot_j @ (rot_i @ center))
            drk -= d_rot_k @ (rot_j @ (rot_i @ center))
        
        grad_x = grad_x.ravel()
        grad_y = grad_y.ravel()
        grad_z = grad_z.ravel()
        grad_ri = grad_x*dri[0] + grad_y*dri[1] + grad_z*dri[2]
        grad_rj = grad_x*drj[0] + grad_y*drj[1] + grad_z*drj[2]
        grad_rk = grad_x*drk[0] + grad_y*drk[1] + grad_z*drk[2]
        return np.vstack([grad_ri, grad_rj, grad_rk]).T

    def _derivative_translation(self, x, grad_x, grad_y, grad_z):
        return np.vstack([grad_x.ravel(), grad_y.ravel(), grad_z.ravel()]).T

    def derivative(self, x, to=["b"]):

        # we expect the input as flattened array, so we reshape it
        x = x.reshape(self.im_shape)
        grad_x, grad_y, grad_z = grad_affine_warp_3D(
            x,
            self.A,
            self.b,
            indexing=self.indexing
        )

        if "rotation" in to or "rot" in to or "scale" in to or "zoom" in to:
            co_x, co_y, co_z = np.meshgrid(
                np.arange(self.im_shape[0]),
                np.arange(self.im_shape[1]),
                np.arange(self.im_shape[2]),
                indexing="ij"
            )
            if self.indexing == "xy":
                co_x = co_x.T
                co_y = co_y.T
                co_z = co_z.T

        derivatives = []
        for var in to:
            if var == "scale":
                derivatives.append(self._derivative_scale(
                    x,
                    grad_x,
                    grad_y,
                    grad_z,
                    co_x,
                    co_y,
                    co_z
                ))
            elif var == "zoom":
                derivatives.append(self._derivative_zoom(
                    x,
                    grad_x,
                    grad_y,
                    grad_z,
                    co_x,
                    co_y,
                    co_z
                ))
            elif var in ["rot", "rotation"]:
                derivatives.append(self._derivative_rotation(
                    x,
                    grad_x,
                    grad_y,
                    grad_z,
                    co_x,
                    co_y,
                    co_z
                ))
            elif var in ["trans", "translation", "b"]:
                derivatives.append(self._derivative_translation(
                    x,
                    grad_x,
                    grad_y,
                    grad_z
                ))
            else:
                derivatives.append(np.zeros((x.size, 0)))
        return derivatives