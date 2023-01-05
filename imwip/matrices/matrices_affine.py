"""
:file:      matrices_affine.py
:brief:     Functions for generating sparse matrix representations
            of image warps described by an affine transformation.
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


import os
import numpy as np
from math import prod
from scipy.sparse import coo_matrix
from .matrix_kernels import *


path = os.path.dirname(__file__)
cubic_2D_coefficients = np.loadtxt(path+"/../cpp_backend/cubic_2D_coefficients.inc", delimiter= ",", dtype=np.float32)
cubic_3D_coefficients = np.loadtxt(path+"/../cpp_backend/cubic_3D_coefficients.inc", delimiter= ",", dtype=np.float32)


def affine_warp_matrix(im_shape, A, b, degree=3):
    """
    Generates a sparse matrix representing an image warping operator described by an
    affine transformation.

    :param A: Matrix A of the affine transformation Ax + b.
        It shoud be an 2x2 for 2D warp and 3x3 for a 3D warp.
    :param b: Vector b of the affine transformation Ax + b. It should have length 2 for
        a 2D warp and length 3 for a 3D warp.
    :param degree: Degree of the splines used for interpolation, defaults to 3.

    :type A: :class:`numpy.ndarray`
    :type b: :class:`numpy.ndarray`
    :type degree: 1 or 3, optional

    :return: A matrix representing the warping operator
    :rtype: :class:`scipy.sparse.coo_matrix`
    """

    dim = b.size
    if dim not in [2, 3]:
        raise ValueError("b should be of length 2 or 3")
    im_size = prod(im_shape)

    if dim == 2:
        coeffs = cubic_2D_coefficients
        threads_per_block = (16, 16)
        num_blocks = ((im_shape[0] + 15)//16, (im_shape[1] + 15)//16)

        if degree==1:
            data = np.zeros((im_size, 4), dtype=np.float32)
            coords = np.zeros((2, im_size, 4), dtype=np.int32)
            affine_linear_warp_2D_matrix_kernel[num_blocks, threads_per_block](
                im_shape,
                A,
                b,
                data,
                coords
            )
        elif degree==3:
            data = np.zeros((im_size, 16), dtype=np.float32)
            coords = np.zeros((2, im_size, 16), dtype=np.int32)
            affine_cubic_warp_2D_matrix_kernel[num_blocks, threads_per_block](
                im_shape,
                A,
                b,
                coeffs,
                data,
                coords
            )
        else:
            raise NotImplementedError("Only degree 1 and 3 are implemented.")
    else:
        coeffs = cubic_3D_coefficients
        threads_per_block = (8, 8, 8)
        num_blocks = ((im_shape[0] + 7)//8, (im_shape[1] + 7)//8, (im_shape[2] + 7)//8)

        if degree==1:
            data = np.zeros((im_size, 8), dtype=np.float32)
            coords = np.zeros((2, im_size, 8), dtype=np.int32)
            affine_linear_warp_3D_matrix_kernel[num_blocks, threads_per_block](
                im_shape,
                A,
                b,
                data,
                coords
            )
        elif degree==3:
            data = np.zeros((im_size, 64), dtype=np.float32)
            coords = np.zeros((2, im_size, 64), dtype=np.int32)
            affine_cubic_warp_3D_matrix_kernel[num_blocks, threads_per_block](
                im_shape,
                A,
                b,
                coeffs,
                data,
                coords
            )
        else:
            raise NotImplementedError("Only degree 1 and 3 are implemented.")

    return coo_matrix((data.ravel(), coords.reshape(2, -1)), shape=(im_size, im_size))
