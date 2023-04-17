"""
:file:      warp_algorithms_affine.py
:brief:     Affine warping algorithms using numba kernels
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
from .warp_kernels_affine import (
    affine_linear_warp_2D_kernel,
    affine_cubic_warp_2D_kernel,
    affine_linear_warp_3D_kernel,
    affine_cubic_warp_3D_kernel,
    affine_cubic_warp_3D_kernel_mul
)
import os

path = os.path.dirname(__file__)
cubic_2D_coefficients = np.loadtxt(path+"/../cpp_backend/cubic_2D_coefficients.inc", delimiter= ",", dtype=np.float32)
cubic_2D_coefficients_dx = np.loadtxt(path+"/../cpp_backend/cubic_2D_coefficients_dx.inc", delimiter= ",", dtype=np.float32)
cubic_2D_coefficients_dy = np.loadtxt(path+"/../cpp_backend/cubic_2D_coefficients_dy.inc", delimiter= ",", dtype=np.float32)

cubic_3D_coefficients = np.loadtxt(path+"/../cpp_backend/cubic_3D_coefficients.inc", delimiter= ",", dtype=np.float32)
cubic_3D_coefficients_dx = np.loadtxt(path+"/../cpp_backend/cubic_3D_coefficients_dx.inc", delimiter= ",", dtype=np.float32)
cubic_3D_coefficients_dy = np.loadtxt(path+"/../cpp_backend/cubic_3D_coefficients_dy.inc", delimiter= ",", dtype=np.float32)
cubic_3D_coefficients_dz = np.loadtxt(path+"/../cpp_backend/cubic_3D_coefficients_dz.inc", delimiter= ",", dtype=np.float32)


__all__ = [
    'affine_warp_2D',
    'adjoint_affine_warp_2D',
    'diff_affine_warp_2D',
    'affine_warp_3D',
    'adjoint_affine_warp_3D',
    'diff_affine_warp_3D'
]


def affine_warp_2D(
        f,
        A,
        b,
        f_warped=None,
        degree=3,
        indexing="ij"
    ):
    if f_warped is None:
        f_warped = np.zeros(f.shape, dtype=f.dtype)

    if indexing == "xy":
        A = np.fliplr(np.flipud(A)).copy()
        b = np.flip(b).copy()

    coeffs = cubic_2D_coefficients
    threads_per_block = (16, 16)
    num_blocks = ((f.shape[0] + 15)//16, (f.shape[1] + 15)//16)
    if degree == 1:
        affine_linear_warp_2D_kernel[num_blocks, threads_per_block](
            f,
            A,
            b,
            f_warped,
            False
        )
    elif degree == 3:
        affine_cubic_warp_2D_kernel[num_blocks, threads_per_block](
            f,
            A,
            b,
            f_warped,
            coeffs,
            False
        )
    else:
        raise NotImplementedError("Only degree 1 and 3 are implemented.")

    return f_warped


def adjoint_affine_warp_2D(
        f_warped,
        A,
        b,
        f=None,
        degree=3,
        indexing="ij"
    ):
    if f is None:
        f = np.zeros(f_warped.shape, dtype=f_warped.dtype)

    if indexing == "xy":
        A = np.fliplr(np.flipud(A)).copy()
        b = np.flip(b).copy()

    coeffs = cubic_2D_coefficients
    threads_per_block = (16, 16)
    num_blocks = ((f.shape[0] + 15)//16, (f.shape[1] + 15)//16)
    if degree == 1:
        affine_linear_warp_2D_kernel[num_blocks, threads_per_block](
            f_warped,
            A,
            b,
            f,
            True
        )
    elif degree == 3:
        affine_cubic_warp_2D_kernel[num_blocks, threads_per_block](
            f_warped,
            A,
            b,
            f,
            coeffs,
            True
        )
    else:
        raise NotImplementedError("Only degree 1 and 3 are implemented.")
    return f


def diff_affine_warp_2D(
        f,
        A,
        b,
        diff_x=None,
        diff_y=None,
        indexing="ij"
    ):
    if diff_x is None:
        diff_x = np.zeros(f.shape, dtype=f.dtype)
    if diff_y is None:
        diff_y = np.zeros(f.shape, dtype=f.dtype)

    if indexing == "xy":
        A = np.fliplr(np.flipud(A)).copy()
        b = np.flip(b).copy()

    coeffs_dx = cubic_2D_coefficients_dx
    coeffs_dy = cubic_2D_coefficients_dy
    threads_per_block = (16, 16)
    num_blocks = ((f.shape[0] + 15)//16, (f.shape[1] + 15)//16)
    affine_cubic_warp_2D_kernel[num_blocks, threads_per_block](
        f,
        A,
        b,
        diff_x,
        coeffs_dx,
        False
    )
    affine_cubic_warp_2D_kernel[num_blocks, threads_per_block](
        f,
        A,
        b,
        diff_y,
        coeffs_dy,
        False
    )
    return diff_x, diff_y


def affine_warp_3D(
        f,
        A,
        b,
        f_warped=None,
        degree=3,
        indexing="ij"
    ):
    if f_warped is None:
        f_warped = np.zeros(f.shape, dtype=f.dtype)

    if indexing == "xy":
        A = np.fliplr(np.flipud(A)).copy()
        b = np.flip(b).copy()

    coeffs = cubic_3D_coefficients
    threads_per_block = (8, 8, 8)
    num_blocks = ((f.shape[0] + 7)//8, (f.shape[1] + 7)//8, (f.shape[2] + 7)//8)

    if degree == 1:
        affine_linear_warp_3D_kernel[num_blocks, threads_per_block](
            f,
            A,
            b,
            f_warped,
            False
        )
    elif degree == 3:
        affine_cubic_warp_3D_kernel[num_blocks, threads_per_block](
            f,
            A,
            b,
            f_warped,
            coeffs,
            False
        )
    else:
        raise NotImplementedError("Only degree 1 and 3 are implemented.")

    return f_warped


def adjoint_affine_warp_3D(
        f_warped,
        A,
        b,
        f=None,
        degree=3,
        indexing="ij"
    ):
    if f is None:
        f = np.zeros(f_warped.shape, dtype=f_warped.dtype)

    if indexing == "xy":
        A = np.fliplr(np.flipud(A)).copy()
        b = np.flip(b).copy()

    coeffs = cubic_3D_coefficients
    threads_per_block = (8, 8, 8)
    num_blocks = ((f.shape[0] + 7)//8, (f.shape[1] + 7)//8, (f.shape[2] + 7)//8)
    if degree == 1:
        affine_linear_warp_3D_kernel[num_blocks, threads_per_block](
            f_warped,
            A,
            b,
            f,
            True
        )
    elif degree == 3:
        affine_cubic_warp_3D_kernel[num_blocks, threads_per_block](
            f_warped,
            A,
            b,
            f,
            coeffs,
            True
        )
    else:
        raise NotImplementedError("Only degree 1 and 3 are implemented.")

    return f


def diff_affine_warp_3D(
        f,
        A,
        b,
        diff_x=None,
        diff_y=None,
        diff_z=None,
        indexing="ij"
    ):
    if diff_x is None:
        diff_x = np.zeros(f.shape, dtype=f.dtype)
    if diff_y is None:
        diff_y = np.zeros(f.shape, dtype=f.dtype)
    if diff_z is None:
        diff_z = np.zeros(f.shape, dtype=f.dtype)

    if indexing == "xy":
        A = np.fliplr(np.flipud(A)).copy()
        b = np.flip(b).copy()

    coeffs_dx = cubic_3D_coefficients_dx
    coeffs_dy = cubic_3D_coefficients_dy
    coeffs_dz = cubic_3D_coefficients_dz
    threads_per_block = (8, 8, 8)
    num_blocks = ((f.shape[0] + 7)//8, (f.shape[1] + 7)//8, (f.shape[2] + 7)//8)
    
    if len(f.shape)==4:
        affine_kernel = affine_cubic_warp_3D_kernel_mul
    elif len(f.shape)==3:
        affine_kernel = affine_cubic_warp_3D_kernel
    else:
        raise ValueError("number of f dimensions not supported")
    
    affine_kernel[num_blocks, threads_per_block](
        f,
        A,
        b,
        diff_x,
        coeffs_dx,
        False
    )
    affine_kernel[num_blocks, threads_per_block](
        f,
        A,
        b,
        diff_y,
        coeffs_dy,
        False
    )
    affine_kernel[num_blocks, threads_per_block](
        f,
        A,
        b,
        diff_z,
        coeffs_dz,
        False
    )
    return diff_x, diff_y, diff_z