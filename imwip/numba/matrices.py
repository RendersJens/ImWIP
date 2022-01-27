"""
:file:      matrices.py
:brief:     functions for generating sparse matrix representations
            of image warps
:date:      20 DEC 2021
:author:    Jens Renders
            imec-Visionlab
            University of Antwerp
            jens.renders@uantwerpen.be
"""


import numpy as np
from scipy.sparse import coo_matrix
from numba import cuda
from numba.types import void, int32, float32, boolean, UniTuple
import math
import os

path = os.path.dirname(__file__)
cubic_2D_coefficients = np.loadtxt(path+"/../cuda/cubic_2D_coefficients.inc", delimiter= ",", dtype=np.float32)
cubic_3D_coefficients = np.loadtxt(path+"/../cuda/cubic_3D_coefficients.inc", delimiter= ",", dtype=np.float32)


@cuda.jit(void(
    UniTuple(int32, 2),
    float32[:, ::1],
    float32[:, ::1],
    float32[:, ::1],
    int32[:, :, ::1]
    ))
def linear_warp_2D_matrix_kernel(im_shape, u, v, data, coords):
    i, j = cuda.grid(2)

    if i < im_shape[0] and j < im_shape[1]:
        f_i = float32(i)
        f_j = float32(j)

        # position at which to iterpolate
        x = f_i + u[i, j]
        y = f_j + v[i, j]

        # points from which to interpolate
        x1 = int32(math.floor(x))
        y1 = int32(math.floor(y))
        x2 = int32(x1 + 1)
        y2 = int32(y1 + 1)
        Q = ((x1, y1),
             (x1, y2),
             (x2, y1),
             (x2, y2))

        # interpolation coefficients
        coefficients = ((x2 - x)*(y2 - y),
                        (x2 - x)*(y - y1),
                        (x - x1)*(y2 - y),
                        (x - x1)*(y - y1))


        for m in range(4):
            if 0 <= Q[m][0] < im_shape[0] and 0 <= Q[m][1] < im_shape[1]:
                data[i*im_shape[1] + j, m] = coefficients[m]
                coords[0, i*im_shape[1] + j, m] = i*im_shape[1] + j
                coords[1, i*im_shape[1] + j, m] = Q[m][0]*im_shape[1] + Q[m][1]


@cuda.jit(void(
    UniTuple(int32, 2),
    float32[:, ::1],
    float32[:, ::1],
    float32[::1],
    float32[:, ::1],
    int32[:, :, ::1]
    ))
def cubic_warp_2D_matrix_kernel(im_shape, u, v, coeffs, data, coords):
    i, j = cuda.grid(2)

    if i < im_shape[0] and j < im_shape[1]:
        f_i = float32(i)
        f_j = float32(j)

        # position at which to iterpolate
        x = f_i + u[i, j]
        y = f_j + v[i, j]

        # points from which to interpolate
        x1 = int32(math.floor(x))
        y1 = int32(math.floor(y))
        # xi = x1 - 1 + i

        # interpolation coefficients
        xmx1 = x - float32(x1)
        ymy1 = y - float32(y1)
        x_powers = (float32(1), xmx1, xmx1**2, xmx1**3)
        y_powers = (float32(1), ymy1, ymy1**2, ymy1**3)
        monomials = cuda.local.array(16, float32)
        for py in range(4):
            for px in range(4):
                monomials[py*4 + px] = x_powers[px] * y_powers[py]

        m = 0
        for ii in range(4):
            for jj in range(4):
                Q0 = x1 + ii - 1
                Q1 = y1 + jj - 1
                if 0 <= Q0 < im_shape[0] and 0 <= Q1 < im_shape[1]:
                    coefficient = float32(0)
                    for n in range(16):
                        coefficient += coeffs[m*16 + n] * monomials[n]
                    data[i*im_shape[1] + j, m] = coefficient
                    coords[0, i*im_shape[1] + j, m] = i*im_shape[1] + j
                    coords[1, i*im_shape[1] + j, m] = Q0*im_shape[1] + Q1

                m += 1


def warp_2D_matrix(im_shape, u, v, degree=3):
    im_size = im_shape[0] * im_shape[1]
    coeffs = cubic_2D_coefficients
    threads_per_block = (16, 16)
    num_blocks = ((im_shape[0] + 15)//16, (im_shape[1] + 15)//16)

    if degree==1:
        data = np.zeros((im_size, 4), dtype=np.float32)
        coords = np.zeros((2, im_size, 4), dtype=np.int32)
        linear_warp_2D_matrix_kernel[num_blocks, threads_per_block](
            im_shape,
            u,
            v,
            data,
            coords
        )
    elif degree==3:
        data = np.zeros((im_size, 16), dtype=np.float32)
        coords = np.zeros((2, im_size, 16), dtype=np.int32)
        cubic_warp_2D_matrix_kernel[num_blocks, threads_per_block](
            im_shape,
            u,
            v,
            coeffs,
            data,
            coords
        )
    else:
        raise NotImplementedError("Only degree 1 and 3 are implemented.")

    return coo_matrix((data.ravel(), coords.reshape(2, -1)), shape=(im_size, im_size)).tocsr()



@cuda.jit(void(
    UniTuple(int32, 2),
    float32[:, ::1],
    float32[::1],
    float32[:, ::1],
    int32[:, :, ::1]
    ))
def affine_linear_warp_2D_matrix_kernel(im_shape, A, b, data, coords):
    i, j = cuda.grid(2)

    if i < im_shape[0] and j < im_shape[1]:
        f_i = float32(i)
        f_j = float32(j)

        # position at which to iterpolate
        x = A[0, 0]*f_i + A[0, 1]*f_j + b[0]
        y = A[1, 0]*f_i + A[1, 1]*f_j + b[1]

        # points from which to interpolate
        x1 = int32(math.floor(x))
        y1 = int32(math.floor(y))
        x2 = int32(x1 + 1)
        y2 = int32(y1 + 1)
        Q = ((x1, y1),
             (x1, y2),
             (x2, y1),
             (x2, y2))

        # interpolation coefficients
        coefficients = ((x2 - x)*(y2 - y),
                        (x2 - x)*(y - y1),
                        (x - x1)*(y2 - y),
                        (x - x1)*(y - y1))


        for m in range(4):
            if 0 <= Q[m][0] < im_shape[0] and 0 <= Q[m][1] < im_shape[1]:
                data[i*im_shape[1] + j, m] = coefficients[m]
                coords[0, i*im_shape[1] + j, m] = i*im_shape[1] + j
                coords[1, i*im_shape[1] + j, m] = Q[m][0]*im_shape[1] + Q[m][1]


@cuda.jit(void(
    UniTuple(int32, 2),
    float32[:, ::1],
    float32[::1],
    float32[::1],
    float32[:, ::1],
    int32[:, :, ::1]
    ))
def affine_cubic_warp_2D_matrix_kernel(im_shape, A, b, coeffs, data, coords):
    i, j = cuda.grid(2)

    if i < im_shape[0] and j < im_shape[1]:
        f_i = float32(i)
        f_j = float32(j)

        # position at which to iterpolate
        x = A[0, 0]*f_i + A[0, 1]*f_j + b[0]
        y = A[1, 0]*f_i + A[1, 1]*f_j + b[1]

        # points from which to interpolate
        x1 = int32(math.floor(x))
        y1 = int32(math.floor(y))
        # xi = x1 - 1 + i

        # interpolation coefficients
        xmx1 = x - float32(x1)
        ymy1 = y - float32(y1)
        x_powers = (float32(1), xmx1, xmx1**2, xmx1**3)
        y_powers = (float32(1), ymy1, ymy1**2, ymy1**3)
        monomials = cuda.local.array(16, float32)
        for py in range(4):
            for px in range(4):
                monomials[py*4 + px] = x_powers[px] * y_powers[py]

        m = 0
        for ii in range(4):
            for jj in range(4):
                Q0 = x1 + ii - 1
                Q1 = y1 + jj - 1
                if 0 <= Q0 < im_shape[0] and 0 <= Q1 < im_shape[1]:
                    coefficient = float32(0)
                    for n in range(16):
                        coefficient += coeffs[m*16 + n] * monomials[n]
                    data[i*im_shape[1] + j, m] = coefficient
                    coords[0, i*im_shape[1] + j, m] = i*im_shape[1] + j
                    coords[1, i*im_shape[1] + j, m] = Q0*im_shape[1] + Q1

                m += 1


def affine_warp_2D_matrix(im_shape, A, b, degree=3):
    im_size = im_shape[0] * im_shape[1]
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

    return coo_matrix((data.ravel(), coords.reshape(2, -1)), shape=(im_size, im_size)).tocsr()


@cuda.jit(void(
    UniTuple(int32, 3),
    float32[:, :, ::1],
    float32[:, :, ::1],
    float32[:, :, ::1],
    float32[:, ::1],
    int32[:, :, ::1]
    ))
def linear_warp_3D_matrix_kernel(im_shape, u, v, w, data, coords):
    i, j, k = cuda.grid(3)

    if i < im_shape[0] and j < im_shape[1] and k < im_shape[2]:
        f_i = float32(i)
        f_j = float32(j)
        f_k = float32(k)

        # position at which to iterpolate
        x = f_i + u[i, j, k]
        y = f_j + v[i, j, k]
        z = f_k + w[i, j, k]

        # points from which to interpolate
        x1 = int32(math.floor(x))
        y1 = int32(math.floor(y))
        z1 = int32(math.floor(z))
        x2 = int32(x1 + 1)
        y2 = int32(y1 + 1)
        z2 = int32(z1 + 1)
        Q = ((x1, y1, z1),
             (x2, y1, z1),
             (x1, y2, z1),
             (x2, y2, z1),
             (x1, y1, z2),
             (x2, y1, z2),
             (x1, y2, z2),
             (x2, y2, z2))

        # interpolation coefficients
        coefficients = ((x2 - x)*(y2 - y)*(z2 - z),
                        (x - x1)*(y2 - y)*(z2 - z),
                        (x2 - x)*(y - y1)*(z2 - z),
                        (x - x1)*(y - y1)*(z2 - z),
                        (x2 - x)*(y2 - y)*(z - z1),
                        (x - x1)*(y2 - y)*(z - z1),
                        (x2 - x)*(y - y1)*(z - z1),
                        (x - x1)*(y - y1)*(z - z1))


        for m in range(8):
            if 0 <= Q[m][0] < im_shape[0] and 0 <= Q[m][1] < im_shape[1] and 0 <= Q[m][2] < im_shape[2]:
                data[i*im_shape[1]*im_shape[2] + j*im_shape[1] + k, m] = coefficients[m]
                coords[0, i*im_shape[1]*im_shape[2] + j*im_shape[1] + k, m] = i*im_shape[1]*im_shape[2] + j*im_shape[1] + k
                coords[1, i*im_shape[1]*im_shape[2] + j*im_shape[1] + k, m] = Q[m][0]*im_shape[1]*im_shape[2] + Q[m][1]*im_shape[1] + Q[m][2]


@cuda.jit(void(
    UniTuple(int32, 3),
    float32[:, :, ::1],
    float32[:, :, ::1],
    float32[:, :, ::1],
    float32[::1],
    float32[:, ::1],
    int32[:, :, ::1]
    ))
def cubic_warp_3D_matrix_kernel(im_shape, u, v, w, coeffs, data, coords):
    i, j, k = cuda.grid(3)

    if i < im_shape[0] and j < im_shape[1] and k < im_shape[2]:
        f_i = float32(i)
        f_j = float32(j)
        f_k = float32(k)

        # position at which to iterpolate
        x = f_i + u[i, j, k]
        y = f_j + v[i, j, k]
        z = f_k + w[i, j, k]

        # points from which to interpolate
        x1 = int32(math.floor(x))
        y1 = int32(math.floor(y))
        z1 = int32(math.floor(z))
        # xi = x1 - 1 + i

        # interpolation coefficients
        xmx1 = x - float32(x1)
        ymy1 = y - float32(y1)
        zmz1 = z - float32(z1)
        x_powers = (float32(1), xmx1, xmx1**2, xmx1**3)
        y_powers = (float32(1), ymy1, ymy1**2, ymy1**3)
        z_powers = (float32(1), zmz1, zmz1**2, zmz1**3)
        monomials = cuda.local.array(64, float32)
        for pz in range(4):
            for py in range(4):
                for px in range(4):
                    monomials[pz*16 + py*4 + px] = x_powers[px] * y_powers[py] * z_powers[pz]

        m = 0
        for ii in range(4):
            for jj in range(4):
                for kk in range(4):
                    Q0 = x1 + ii - 1
                    Q1 = y1 + jj - 1
                    Q2 = z1 + kk - 1
                    if 0 <= Q0 < im_shape[0] and 0 <= Q1 < im_shape[1] and 0 <= Q2 < im_shape[2]:
                        coefficient = float32(0)
                        for n in range(64):
                            coefficient += coeffs[m*64 + n] * monomials[n]
                        data[i*im_shape[1]*im_shape[2] + j*im_shape[1] + k, m] = coefficient
                        coords[0, i*im_shape[1]*im_shape[2] + j*im_shape[1] + k, m] = i*im_shape[1]*im_shape[2] + j*im_shape[1] + k
                        coords[1, i*im_shape[1]*im_shape[2] + j*im_shape[1] + k, m] = Q0*im_shape[1]*im_shape[2] + Q1*im_shape[1] + Q2
                    m += 1


def warp_3D_matrix(im_shape, u, v, w, degree=3):
    im_size = im_shape[0] * im_shape[1] * im_shape[2]
    coeffs = cubic_3D_coefficients
    threads_per_block = (8, 8, 8)
    num_blocks = ((im_shape[0] + 7)//8, (im_shape[1] + 7)//8, (im_shape[2] + 7)//8)

    if degree==1:
        data = np.zeros((im_size, 8), dtype=np.float32)
        coords = np.zeros((2, im_size, 8), dtype=np.int32)
        linear_warp_3D_matrix_kernel[num_blocks, threads_per_block](
            im_shape,
            u,
            v,
            w,
            data,
            coords
        )
    elif degree==3:
        data = np.zeros((im_size, 64), dtype=np.float32)
        coords = np.zeros((2, im_size, 64), dtype=np.int32)
        cubic_warp_3D_matrix_kernel[num_blocks, threads_per_block](
            im_shape,
            u,
            v,
            w,
            coeffs,
            data,
            coords
        )
    else:
        raise NotImplementedError("Only degree 1 and 3 are implemented.")

    return coo_matrix((data.ravel(), coords.reshape(2, -1)), shape=(im_size, im_size)).tocsr()



@cuda.jit(void(
    UniTuple(int32, 3),
    float32[:, ::1],
    float32[::1],
    float32[:, ::1],
    int32[:, :, ::1]
    ))
def affine_linear_warp_3D_matrix_kernel(im_shape, A, b, data, coords):
    i, j, k = cuda.grid(3)

    if i < im_shape[0] and j < im_shape[1] and k < im_shape[2]:
        f_i = float32(i)
        f_j = float32(j)
        f_k = float32(k)

        # position at which to iterpolate
        x = A[0, 0]*f_i + A[0, 1]*f_j + A[0, 2]*f_k + b[0]
        y = A[1, 0]*f_i + A[1, 1]*f_j + A[1, 2]*f_k + b[1]
        z = A[2, 0]*f_i + A[2, 1]*f_j + A[2, 2]*f_k + b[2]

        # points from which to interpolate
        x1 = int32(math.floor(x))
        y1 = int32(math.floor(y))
        z1 = int32(math.floor(z))
        x2 = int32(x1 + 1)
        y2 = int32(y1 + 1)
        z2 = int32(z1 + 1)
        Q = ((x1, y1, z1),
             (x2, y1, z1),
             (x1, y2, z1),
             (x2, y2, z1),
             (x1, y1, z2),
             (x2, y1, z2),
             (x1, y2, z2),
             (x2, y2, z2))

        # interpolation coefficients
        coefficients = ((x2 - x)*(y2 - y)*(z2 - z),
                        (x - x1)*(y2 - y)*(z2 - z),
                        (x2 - x)*(y - y1)*(z2 - z),
                        (x - x1)*(y - y1)*(z2 - z),
                        (x2 - x)*(y2 - y)*(z - z1),
                        (x - x1)*(y2 - y)*(z - z1),
                        (x2 - x)*(y - y1)*(z - z1),
                        (x - x1)*(y - y1)*(z - z1))


        for m in range(8):
            if 0 <= Q[m][0] < im_shape[0] and 0 <= Q[m][1] < im_shape[1] and 0 <= Q[m][2] < im_shape[2]:
                data[i*im_shape[1]*im_shape[2] + j*im_shape[1] + k, m] = coefficients[m]
                coords[0, i*im_shape[1]*im_shape[2] + j*im_shape[1] + k, m] = i*im_shape[1]*im_shape[2] + j*im_shape[1] + k
                coords[1, i*im_shape[1]*im_shape[2] + j*im_shape[1] + k, m] = Q[m][0]*im_shape[1]*im_shape[2] + Q[m][1]*im_shape[1] + Q[m][2]


@cuda.jit(void(
    UniTuple(int32, 3),
    float32[:, ::1],
    float32[::1],
    float32[::1],
    float32[:, ::1],
    int32[:, :, ::1]
    ))
def affine_cubic_warp_3D_matrix_kernel(im_shape, A, b, coeffs, data, coords):
    i, j, k = cuda.grid(3)

    if i < im_shape[0] and j < im_shape[1] and k < im_shape[2]:
        f_i = float32(i)
        f_j = float32(j)
        f_k = float32(k)

        # position at which to iterpolate
        x = A[0, 0]*f_i + A[0, 1]*f_j + A[0, 2]*f_k + b[0]
        y = A[1, 0]*f_i + A[1, 1]*f_j + A[1, 2]*f_k + b[1]
        z = A[2, 0]*f_i + A[2, 1]*f_j + A[2, 2]*f_k + b[2]

        # points from which to interpolate
        x1 = int32(math.floor(x))
        y1 = int32(math.floor(y))
        z1 = int32(math.floor(z))
        # xi = x1 - 1 + i

        # interpolation coefficients
        xmx1 = x - float32(x1)
        ymy1 = y - float32(y1)
        zmz1 = z - float32(z1)
        x_powers = (float32(1), xmx1, xmx1**2, xmx1**3)
        y_powers = (float32(1), ymy1, ymy1**2, ymy1**3)
        z_powers = (float32(1), zmz1, zmz1**2, zmz1**3)
        monomials = cuda.local.array(64, float32)
        for pz in range(4):
            for py in range(4):
                for px in range(4):
                    monomials[pz*16 + py*4 + px] = x_powers[px] * y_powers[py] * z_powers[pz]

        m = 0
        for ii in range(4):
            for jj in range(4):
                for kk in range(4):
                    Q0 = x1 + ii - 1
                    Q1 = y1 + jj - 1
                    Q2 = z1 + kk - 1
                    if 0 <= Q0 < im_shape[0] and 0 <= Q1 < im_shape[1] and 0 <= Q2 < im_shape[2]:
                        coefficient = float32(0)
                        for n in range(64):
                            coefficient += coeffs[m*64 + n] * monomials[n]
                        data[i*im_shape[1]*im_shape[2] + j*im_shape[1] + k, m] = coefficient
                        coords[0, i*im_shape[1]*im_shape[2] + j*im_shape[1] + k, m] = i*im_shape[1]*im_shape[2] + j*im_shape[1] + k
                        coords[1, i*im_shape[1]*im_shape[2] + j*im_shape[1] + k, m] = Q0*im_shape[1]*im_shape[2] + Q1*im_shape[1] + Q2
                    m += 1


def affine_warp_3D_matrix(im_shape, A, b, degree=3):
    im_size = im_shape[0] * im_shape[1] * im_shape[2]
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

    return coo_matrix((data.ravel(), coords.reshape(2, -1)), shape=(im_size, im_size)).tocsr()
