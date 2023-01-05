"""
:file:      warp_algorithms.py
:brief:     Warping algorithms using numba kernels
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
from .warp_kernels import (
    linear_warp_2D_kernel,
    cubic_warp_2D_kernel,
    linear_warp_3D_kernel,
    cubic_warp_3D_kernel
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
    'warp_2D',
    'adjoint_warp_2D',
    'diff_warp_2D',
    'warp_3D',
    'adjoint_warp_3D',
    'diff_warp_3D'
]


def warp_2D(
        f,
        u,
        v,
        f_warped=None,
        degree=3,
    ):
    if f_warped is None:
        f_warped = np.zeros(f.shape, dtype=f.dtype)

    coeffs = cubic_2D_coefficients
    threads_per_block = (16, 16)
    num_blocks = ((f.shape[0] + 15)//16, (f.shape[1] + 15)//16)
    if degree == 1:
        linear_warp_2D_kernel[num_blocks, threads_per_block](
            f,
            u,
            v,
            f_warped,
            False
        )
    elif degree == 3:
        cubic_warp_2D_kernel[num_blocks, threads_per_block](
            f,
            u,
            v,
            f_warped,
            coeffs,
            False
        )
    else:
        raise NotImplementedError("Only degree 1 and 3 are implemented.")

    return f_warped


def adjoint_warp_2D(
        f_warped,
        u,
        v,
        f=None,
        degree=3
    ):
    if f is None:
        f = np.zeros(f_warped.shape, dtype=f_warped.dtype)

    coeffs = cubic_2D_coefficients
    threads_per_block = (16, 16)
    num_blocks = ((f.shape[0] + 15)//16, (f.shape[1] + 15)//16)
    if degree == 1:
        linear_warp_2D_kernel[num_blocks, threads_per_block](
            f_warped,
            u,
            v,
            f,
            True
        )
    elif degree == 3:
        cubic_warp_2D_kernel[num_blocks, threads_per_block](
            f_warped,
            u,
            v,
            f,
            coeffs,
            True
        )
    else:
        raise NotImplementedError("Only degree 1 and 3 are implemented.")
    return f


def diff_warp_2D(
        f,
        u,
        v
    ):
    diff_x = np.zeros(f.shape, dtype=f.dtype)
    diff_y = np.zeros(f.shape, dtype=f.dtype)

    coeffs_dx = cubic_2D_coefficients_dx
    coeffs_dy = cubic_2D_coefficients_dy
    threads_per_block = (16, 16)
    num_blocks = ((f.shape[0] + 15)//16, (f.shape[1] + 15)//16)
    cubic_warp_2D_kernel[num_blocks, threads_per_block](
        f,
        u,
        v,
        diff_x,
        coeffs_dx,
        False
    )
    cubic_warp_2D_kernel[num_blocks, threads_per_block](
        f,
        u,
        v,
        diff_y,
        coeffs_dy,
        False
    )
    return diff_x, diff_y


def warp_3D(
        f,
        u,
        v,
        w,
        f_warped=None,
        degree=3
    ):
    if f_warped is None:
        f_warped = np.zeros(f.shape, dtype=f.dtype)

    coeffs = cubic_3D_coefficients
    threads_per_block = (8, 8, 8)
    num_blocks = ((f.shape[0] + 7)//8, (f.shape[1] + 7)//8, (f.shape[2] + 7)//8)

    if degree == 1:
        linear_warp_3D_kernel[num_blocks, threads_per_block](
            f,
            u,
            v,
            w,
            f_warped,
            False
        )
    elif degree == 3:
        cubic_warp_3D_kernel[num_blocks, threads_per_block](
            f,
            u,
            v,
            w,
            f_warped,
            coeffs,
            False
        )
    else:
        raise NotImplementedError("Only degree 1 and 3 are implemented.")

    return f_warped


def adjoint_warp_3D(
        f_warped,
        u,
        v,
        w,
        f=None,
        degree=3
    ):
    if f is None:
        f = np.zeros(f_warped.shape, dtype=f_warped.dtype)

    coeffs = cubic_3D_coefficients
    threads_per_block = (8, 8, 8)
    num_blocks = ((f.shape[0] + 7)//8, (f.shape[1] + 7)//8, (f.shape[2] + 7)//8)
    if degree == 1:
        linear_warp_3D_kernel[num_blocks, threads_per_block](
            f_warped,
            u,
            v,
            w,
            f,
            True
        )
    elif degree == 3:
        cubic_warp_3D_kernel[num_blocks, threads_per_block](
            f_warped,
            u,
            v,
            w,
            f,
            coeffs,
            True
        )
    else:
        raise NotImplementedError("Only degree 1 and 3 are implemented.")

    return f


def diff_warp_3D(
        f,
        u,
        v,
        w
    ):
    diff_x = np.zeros(f.shape, dtype=f.dtype)
    diff_y = np.zeros(f.shape, dtype=f.dtype)
    diff_z = np.zeros(f.shape, dtype=f.dtype)

    coeffs_dx = cubic_3D_coefficients_dx
    coeffs_dy = cubic_3D_coefficients_dy
    coeffs_dz = cubic_3D_coefficients_dz
    threads_per_block = (8, 8, 8)
    num_blocks = ((f.shape[0] + 7)//8, (f.shape[1] + 7)//8, (f.shape[2] + 7)//8)
    cubic_warp_3D_kernel[num_blocks, threads_per_block](
        f,
        u,
        v,
        w,
        diff_x,
        coeffs_dx,
        False
    )
    cubic_warp_3D_kernel[num_blocks, threads_per_block](
        f,
        u,
        v,
        w,
        diff_y,
        coeffs_dy,
        False
    )
    cubic_warp_3D_kernel[num_blocks, threads_per_block](
        f,
        u,
        v,
        w,
        diff_z,
        coeffs_dz,
        False
    )
    return diff_x, diff_y, diff_z