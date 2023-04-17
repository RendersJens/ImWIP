"""
:file:      operators_affine.py
:brief:     High level access to the affine warping functions through linear operators.
            These operators also help with differentiation towards affine and rigid parameters.
:author:    Jens Renders
"""

# This file is part of ImWIP.
#
# ImWIP is free software: you can redistribute it and/or modify it under the terms of
# the GNU General Public License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
#
# ImWIP is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of
# the GNU General Public License along with ImWIP. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import imwip
from scipy.sparse.linalg import LinearOperator
from collections.abc import Iterable


class AffineWarpingOperator2D(LinearOperator):
    """
    A 2D warping operator represented as a scipy LinearOperator.
    It is defined by an affine transformation :math:`Ax + b` or other
    affine or rigid parameters such as scaling, rotations and translations.

    :param im_shape: shape of the images to be warped
    :param A: 2x2 matrix A of the affine transformation Ax + b.
    :param b: Vector b of the affine transformation Ax + b. It should have length 2.
    :param scale: Scaling parameters of the transformation. If a float, scaling will be
        equal in all axes.
    :param rotation: Rotation parameter of the transformation, in radians.
    :param translation: Translation parameters of the transformation.
    :param centered: Whether to use the image center as center for the transformation,
        instead of the (0,0) coordinate. Defaults to `True`.
    :param degree: Degree of the splines used for interpolation
    :param adjoint_type: Method to compute the adjoint. Defaults to ``exact``.
    :param derivative_type: Method to compute the derivatives. Defaults to ``exact``.
    :param indexing: ``ij`` uses standard numpy array indexing. "xy" reversed the order of
        the indexes, making the vertical axis the first index. This can be more intuitive
        for 2D arrays. Defaults to ``ij``.
    :param backend: Whether to use the cpp or numba backend. If None, ``cpp`` will be used
        if available, else ``numba``

    :type im_shape: tuple of ints
    :type A: :class:`numpy.ndarray`, optional
    :type b: :class:`numpy.ndarray`, optional
    :type scale: float or sequence of floats, optional
    :type rotation: float, optional
    :type translation: sequence of floats, optional
    :type centered: bool, optional
    :type degree: 1 or 3, optional
    :type adjoint_type: ``exact`` or ``inverse``, optional
    :type derivative_type: ``exact`` or ``approx``, optional
    :type indexing: ``ij`` or ``xy``, optional
    :type backend: ``cpp`` or ``numba``, optional
    """

    def __init__(
            self,
            im_shape,
            A=None,
            b=None,
            scale=None,
            rotation=None,
            translation=None,
            centered=True,
            center=None,
            degree=3,
            adjoint_type="exact",
            derivative_type="exact",
            indexing="ij",
            backend=None
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
        self.degree = degree
        self.adjoint_type = adjoint_type
        self.derivative_type = derivative_type
        if indexing not in ["ij", "xy"]:
            raise ValueError('indexing should be "xy" or "ij"')
        self.indexing = indexing
        self.backend = backend

    def _matvec(self, x):

        # we expect the input as flattened array, so we reshape it
        x = x.reshape(self.im_shape)

        # perform the warp
        x_warped = imwip.affine_warp(
            x,
            self.A,
            self.b,
            indexing=self.indexing,
            degree=self.degree,
            backend=self.backend
        )

        # return as flattened array
        return x_warped.ravel()

    def _rmatvec(self, x_warped):
        
        # we expect the input as flattened array, so we reshape it
        x_warped = x_warped.reshape(self.im_shape)

        # perform the adjoint warp
        if self.adjoint_type == "exact":
            x = imwip.adjoint_affine_warp(
                x_warped,
                self.A,
                self.b,
                degree=self.degree,
                indexing=self.indexing,
                backend=self.backend
            )
        elif self.adjoint_type == "inverse":
            Ai = np.linalg.inv(self.A)
            x = imwip.affine_warp(
                x_warped,
                Ai,
                -Ai @ self.b,
                degree=self.degree,
                indexing=self.indexing,
                backend=self.backend
            )
        else:
            raise NotImplementedError("adjoint type should be 'exact' or 'inverse'")
               
        # return as flattened array
        return x.ravel()

    def _derivative_scale(self, x, diff_x, diff_y, co_x, co_y):
        dxds = co_x.astype(np.float32)
        dyds = co_y.astype(np.float32)
        if self.centered:
            dxds -= self.center[0]
            dyds -= self.center[1]
        
        diff_sx = diff_x * dxds
        diff_sy = diff_y * dyds
        return np.vstack([diff_sx.ravel(), diff_sy.ravel()]).T

    def _derivative_zoom(self, x, diff_x, diff_y, co_x, co_y):
        diff_sx, diff_sy = self._derivative_scale(x, diff_x, diff_y, co_x, co_y).T
        diff_s = diff_sx + diff_sy
        return diff_s.reshape((-1, 1))

    def _derivative_rotation(self, x, diff_x, diff_y, co_x, co_y):
        dxdr = -co_x*np.sin(self.rotation) + co_y*np.cos(self.rotation)
        dydr = -co_x*np.cos(self.rotation) - co_y*np.sin(self.rotation)
        if self.centered:
            dxdr += self.center[0]*np.sin(self.rotation) - self.center[1]*np.cos(self.rotation)
            dydr += self.center[0]*np.cos(self.rotation) + self.center[1]*np.sin(self.rotation)

        diff_rotation = diff_x*dxdr + diff_y*dydr
        return diff_rotation.reshape((-1, 1))

    def _derivative_translation(self, x, diff_x, diff_y):
        return np.vstack([diff_x.ravel(), diff_y.ravel()]).T

    def derivative(self, x, to=["b"]):
        """
        Computes the derivative of ``M @ x`` where ``M`` is this operator and ``x``
        is a raveled image. You can differentiate towards any of the available parameters
        for defining the operator, independend of which ones you used to define it. For
        example, you can define the operator using A en b and then
        differentiate it towards the rotation parameter.

        :param x: A raveled image x on which this operator acts.
        :param to: The parameters to differentiate towards.

        :type x: :class:`numpy.ndarray`
        :type to: sequence of strings

        :return: The derivative as a dense matrix
        :rtype: :class:`numpy.ndarray`
        """

        # we expect the input as flattened array, so we reshape it
        x = x.reshape(self.im_shape)
        if self.derivative_type == "exact":
            diff_x, diff_y = imwip.diff_affine_warp(
                x,
                self.A,
                self.b,
                indexing=self.indexing,
                backend=self.backend
            )
        else:
            x_warped = imwip.affine_warp(
                x,
                self.A,
                self.b,
                indexing=self.indexing,
                backend=self.backend
            )
            diff_x, diff_y = np.gradient(x_warped)

        if "rotation" in to or "rot" in to or "scale" in to or "zoom" in to:
            co_x, co_y = np.meshgrid(
                np.arange(self.im_shape[0]),
                np.arange(self.im_shape[1]),
                indexing=self.indexing
            )
        derivatives = []
        for var in to:
            if var == "scale":
                derivatives.append(self._derivative_scale(x, diff_x, diff_y, co_x, co_y))
            elif var == "zoom":
                derivatives.append(self._derivative_zoom(x, diff_x, diff_y, co_x, co_y))
            elif var in ["rot", "rotation"]:
                derivatives.append(self._derivative_rotation(x, diff_x, diff_y, co_x, co_y))
            elif var in ["trans", "translation", "b"]:
                derivatives.append(self._derivative_translation(x, diff_x, diff_y))
            else:
                derivatives.append(np.zeros((x.size, 0)))
        return derivatives


class AffineWarpingOperator3D(LinearOperator):
    """
    A 3D warping operator represented as a scipy LinearOperator.
    It is defined by an affine transformation :math:`Ax + b` or other
    affine or rigid parameters such as scaling, rotations and translations.
    
    :param im_shape: shape of the images to be warped
    :param A: 3x3 matrix A of the affine transformation Ax + b.
    :param b: Vector b of the affine transformation Ax + b. It should have length 3.
    :param scale: Scaling parameters of the transformation. If a float, scaling will be
        equal in all axes.
    :param rotation: Rotation parameters of the transformation, in radians, one for each
        axis. The rotations will be applied in the order of the axes.
    :param cayley: Three Cayley parameters. This is an alternative parametrisation of a
        3D rotation using the Cayley transform, which is numerically more favorable.
        It can be interpreted as the rotation corresponding to the quaternion 1+ix+jy+kz,
        where x, y and z are the Cayley parameters.
    :param translation: Translation parameters of the transformation.
    :param centered: Whether to use the provided center as center for the transformation,
        instead of the (0,0,0) coordinate. Defaults to `True`.
    :param center: The center of the transformation. If None, it will be set to the image
        center. Defaults to None.
    :param degree: Degree of the splines used for interpolation
    :param adjoint_type: Method to compute the adjoint. Defaults to ``exact``.
    :param derivative_type: Method to compute the derivatives. Defaults to ``exact``.
    :param indexing: ``ij`` uses standard numpy array indexing. "xy" reversed the order of
        the indexes.
    :param backend: Whether to use the cpp or numba backend. If None, ``cpp`` will be used
        if available, else ``numba``

    :type im_shape: tuple of ints
    :type A: :class:`numpy.ndarray`, optional
    :type b: :class:`numpy.ndarray`, optional
    :type scale: float or sequence of floats, optional
    :type rotation: sequence of floats, optional
    :type cayley: sequence of floats, optional
    :type translation: sequence of floats, optional
    :type centered: bool, optional
    :type center: :class:`numpy.ndarray`, optional
    :type degree: 1 or 3, optional
    :type adjoint_type: ``exact`` or ``inverse``, optional
    :type derivative_type: ``exact`` or ``approx``, optional
    :type indexing: ``ij`` or ``xy``, optional
    :type backend: ``cpp`` or ``numba``, optional
    """

    def __init__(
            self,
            im_shape,
            A=None, 
            b=None,
            scale=None,
            rotation=None,
            cayley=None,
            translation=None,
            centered=True,
            center=None,
            degree=3,
            adjoint_type="exact",
            indexing="ij",
            backend=None
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
        elif cayley is not None:
            self.cayley = cayley
            u, v, w = self.cayley
            P = np.array([
                [ 0,  u, v],
                [-u,  0, w],
                [-v, -w, 0]
            ], dtype=self.dtype)
            I = np.eye(3)
            self.A = (I - P) @ np.linalg.inv(I + P)
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
        self.degree = degree
        self.adjoint_type = adjoint_type
        if indexing not in ["ij", "xy"]:
            raise ValueError('indexing should be "xy" or "ij"')
        self.indexing = indexing
        self.backend = backend

    def _matvec(self, x):

        # we expect the input as flattened array, so we reshape it
        x = x.reshape(self.im_shape)

        # perform the warp
        x_warped = imwip.affine_warp(
            x,
            self.A,
            self.b,
            degree=self.degree,
            indexing=self.indexing,
            backend=self.backend
        )

        # return as flattened array
        return x_warped.ravel()

    def _rmatvec(self, x_warped):

        # we expect the input as flattened array, so we reshape it
        x_warped = x_warped.reshape(self.im_shape)

        # perform the adjoint warp
        if self.adjoint_type == "exact":
            x = imwip.adjoint_affine_warp(
                x_warped,
                self.A,
                self.b,
                degree=self.degree,
                indexing=self.indexing,
                backend=self.backend
            )
        elif self.adjoint_type == "inverse":
            Ai = np.linalg.inv(self.A)
            x = imwip.affine_warp(
                x_warped,
                Ai,
                -Ai @ self.b,
                degree=self.degree,
                indexing=self.indexing,
                backend=self.backend
            )
        else:
            raise NotImplementedError("adjoint type should be 'exact' or 'inverse'")

        # return as flattened array
        return x.ravel()

    def _derivative_scale(
            self,
            x,
            diff_x,
            diff_y,
            diff_z,
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
        
        diff_sx = diff_x * dxds
        diff_sy = diff_y * dyds
        diff_sz = diff_z * dzds
        return np.vstack([diff_sx.ravel(), diff_sy.ravel(), diff_sz.ravel()]).T

    def _derivative_zoom(
            self,
            x,
            diff_x,
            diff_y,
            diff_z,
            co_x,
            co_y,
            co_z
        ):
        diff_sx, diff_sy, diff_sz = self._derivative_scale(x, diff_x, diff_y, co_x, co_y).T
        diff_s = diff_sx + diff_sy + diff_sz
        return diff_s.reshape((-1, 1))

    def _derivative_rotation(
            self,
            x,
            diff_x,
            diff_y,
            diff_z,
            co_x,
            co_y,
            co_z
        ):
        rotation = [
            np.arctan2(-self.A[2,1], self.A[2,2]),
            np.arctan2(-self.A[2,0], np.hypot(self.A[2,1], self.A[2,2])),
            np.arctan2(-self.A[1,0], self.A[0,0])
        ]
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
        
        diff_x = diff_x.ravel()
        diff_y = diff_y.ravel()
        diff_z = diff_z.ravel()
        diff_ri = diff_x*dri[0] + diff_y*dri[1] + diff_z*dri[2]
        diff_rj = diff_x*drj[0] + diff_y*drj[1] + diff_z*drj[2]
        diff_rk = diff_x*drk[0] + diff_y*drk[1] + diff_z*drk[2]
        return np.vstack([diff_ri, diff_rj, diff_rk]).T

    def _derivative_cayley(
            self,
            x,
            diff_x,
            diff_y,
            diff_z,
            co_x,
            co_y,
            co_z
        ):
        u, v, w = self.cayley

        P = np.array([
            [ 0,  u, v],
            [-u,  0, w],
            [-v, -w, 0]
        ], dtype=self.dtype)

        dP_du = np.array([
            [ 0, 1, 0],
            [-1, 0, 0],
            [ 0, 0, 0]
        ], dtype=self.dtype)
        dP_dv = np.array([
            [ 0, 0, 1],
            [ 0, 0, 0],
            [-1, 0, 0]
        ], dtype=self.dtype)
        dP_dw = np.array([
            [0,  0, 0],
            [0,  0, 1],
            [0, -1, 0]
        ], dtype=self.dtype)

        # (I + P)^-1
        I = np.eye(3)
        IPinv = np.linalg.inv(I + P)

        # derivatives of (I + P)^-1
        dIPinv_du = -IPinv @ dP_du @ IPinv
        dIPinv_dv = -IPinv @ dP_dv @ IPinv
        dIPinv_dw = -IPinv @ dP_dw @ IPinv

        # derivatives of A = (I - P)(I + P)^-1
        dA_du = (I - P) @ dIPinv_du - dP_du @ IPinv
        dA_dv = (I - P) @ dIPinv_dv - dP_dv @ IPinv
        dA_dw = (I - P) @ dIPinv_dw - dP_dw @ IPinv


        co = np.vstack([co_x.ravel(), co_y.ravel(), co_z.ravel()])

        du = dA_du @ co
        dv = dA_dv @ co
        dw = dA_dw @ co

        if self.centered:
            center = self.center.reshape((3, 1))
            du -= dA_du @ center
            dv -= dA_dv @ center
            dw -= dA_dw @ center
        
        diff_x = diff_x.ravel()
        diff_y = diff_y.ravel()
        diff_z = diff_z.ravel()
        diff_u = diff_x*du[0] + diff_y*du[1] + diff_z*du[2]
        diff_v = diff_x*dv[0] + diff_y*dv[1] + diff_z*dv[2]
        diff_w = diff_x*dw[0] + diff_y*dw[1] + diff_z*dw[2]
        return np.vstack([diff_u, diff_v, diff_w]).T

    def _derivative_translation(self, x, diff_x, diff_y, diff_z):
        return np.vstack([diff_x.ravel(), diff_y.ravel(), diff_z.ravel()]).T

    def derivative(self, x, to=["b"]):
        """
        Computes the derivative of ``M @ x`` where ``M`` is this operator and ``x``
        is a raveled image. You can differentiate towards any of the available parameters
        for defining the operator, independend of which ones you used to define it. For
        example, you can define the operator using the rotation parameters and then
        differentiate it towards the caylay parameters.

        :param x: A raveled image x on which this operator acts.
        :param to: The parameters to differentiate towards.

        :type x: :class:`numpy.ndarray`
        :type to: sequence of strings

        :return: The derivative as a dense matrix
        :rtype: :class:`numpy.ndarray`
        """

        # we expect the input as flattened array, so we reshape it
        x = x.reshape(self.im_shape)
        diff_x, diff_y, diff_z = imwip.diff_affine_warp(
            x,
            self.A,
            self.b,
            indexing=self.indexing,
            backend=self.backend
        )

        if "cayley" in to or "rotation" in to or "rot" in to or "scale" in to or "zoom" in to:
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
                    diff_x,
                    diff_y,
                    diff_z,
                    co_x,
                    co_y,
                    co_z
                ))
            elif var == "zoom":
                derivatives.append(self._derivative_zoom(
                    x,
                    diff_x,
                    diff_y,
                    diff_z,
                    co_x,
                    co_y,
                    co_z
                ))
            elif var in ["rot", "rotation"]:
                derivatives.append(self._derivative_rotation(
                    x,
                    diff_x,
                    diff_y,
                    diff_z,
                    co_x,
                    co_y,
                    co_z
                ))
            elif var == "cayley":
                derivatives.append(self._derivative_cayley(
                    x,
                    diff_x,
                    diff_y,
                    diff_z,
                    co_x,
                    co_y,
                    co_z
                ))
            elif var in ["trans", "translation", "b"]:
                derivatives.append(self._derivative_translation(
                    x,
                    diff_x,
                    diff_y,
                    diff_z
                ))
            else:
                derivatives.append(np.zeros((x.size, 0)))
        return derivatives