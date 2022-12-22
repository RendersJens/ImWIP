"""
:file:      warp_algorithms_affine.py
:brief:     Affine warping algorithms using numba kernels
:date:      20 DEC 2021
:author:    Jens Renders
            imec-Visionlab
            University of Antwerp
            jens.renders@uantwerpen.be
"""

import numpy as np
from .warp_kernels_affine import (affine_linear_warp_2D_kernel,
                                  affine_cubic_warp_2D_kernel,
                                  affine_linear_warp_3D_kernel,
                                  affine_cubic_warp_3D_kernel)
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
    'grad_affine_warp_2D',
    'affine_warp_3D',
    'adjoint_affine_warp_3D',
    'grad_affine_warp_3D'
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


def grad_affine_warp_2D(
        f,
        A,
        b,
        indexing="ij"
    ):
    grad_x = np.zeros(f.shape, dtype=f.dtype)
    grad_y = np.zeros(f.shape, dtype=f.dtype)

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
        grad_x,
        coeffs_dx,
        False
    )
    affine_cubic_warp_2D_kernel[num_blocks, threads_per_block](
        f,
        A,
        b,
        grad_y,
        coeffs_dy,
        False
    )
    return grad_x, grad_y


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


def grad_affine_warp_3D(
        f,
        A,
        b,
        indexing="ij"
    ):
    grad_x = np.zeros(f.shape, dtype=f.dtype)
    grad_y = np.zeros(f.shape, dtype=f.dtype)
    grad_z = np.zeros(f.shape, dtype=f.dtype)

    if indexing == "xy":
        A = np.fliplr(np.flipud(A)).copy()
        b = np.flip(b).copy()

    coeffs_dx = cubic_3D_coefficients_dx
    coeffs_dy = cubic_3D_coefficients_dy
    coeffs_dz = cubic_3D_coefficients_dz
    threads_per_block = (8, 8, 8)
    num_blocks = ((f.shape[0] + 7)//8, (f.shape[1] + 7)//8, (f.shape[2] + 7)//8)
    affine_cubic_warp_3D_kernel[num_blocks, threads_per_block](
        f,
        A,
        b,
        grad_x,
        coeffs_dx,
        False
    )
    affine_cubic_warp_3D_kernel[num_blocks, threads_per_block](
        f,
        A,
        b,
        grad_y,
        coeffs_dy,
        False
    )
    affine_cubic_warp_3D_kernel[num_blocks, threads_per_block](
        f,
        A,
        b,
        grad_z,
        coeffs_dz,
        False
    )
    return grad_x, grad_y, grad_z