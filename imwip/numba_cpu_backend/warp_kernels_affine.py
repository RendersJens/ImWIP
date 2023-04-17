"""
:file:      warp_kernels_affine.py
:brief:     Affine warping kernels using numba
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
from numba import njit, prange
from numba.types import int32, float32
import math


@njit(parallel=True, fastmath=True)
def affine_cubic_warp_3D_kernel(f, A, b, f_warped, coeffs):
    for i in prange(f.shape[0]):
        for j in prange(f.shape[1]):
            for k in prange(f.shape[2]):
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
                monomials = np.zeros(64, dtype=np.float32)
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
                            if 0 <= Q0 < f.shape[0] and 0 <= Q1 < f.shape[1] and 0 <= Q2 < f.shape[2]:
                                coefficient = float32(0)
                                for n in range(64):
                                    coefficient += coeffs[m*64 + n] * monomials[n]
                                f_warped[i, j, k] += coefficient * f[Q0, Q1, Q2]
                            m += 1

@njit(parallel=True, fastmath=True)
def affine_cubic_warp_3D_kernel_mul(f, A, b, f_warped, coeffs):
    for i in prange(f.shape[0]):
        for j in prange(f.shape[1]):
            for k in prange(f.shape[2]):
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
                monomials = np.zeros(64, dtype=np.float32)
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
                            if 0 <= Q0 < f.shape[0] and 0 <= Q1 < f.shape[1] and 0 <= Q2 < f.shape[2]:
                                coefficient = float32(0)
                                for n in range(64):
                                    coefficient += coeffs[m*64 + n] * monomials[n]                                   
                                for l in range(f.shape[3]):
                                    f_warped[i, j, k, l] += coefficient * f[Q0, Q1, Q2, l]
                            m += 1

@njit(fastmath=True)
def adjoint_affine_cubic_warp_3D_kernel(f, A, b, f_warped, coeffs):
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            for k in range(f.shape[2]):
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
                monomials = np.zeros(64, dtype=np.float32)
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
                            if 0 <= Q0 < f.shape[0] and 0 <= Q1 < f.shape[1] and 0 <= Q2 < f.shape[2]:
                                coefficient = float32(0)
                                for n in range(64):
                                    coefficient += coeffs[m*64 + n] * monomials[n]
                                f_warped[Q0, Q1, Q2] += coefficient * f[i, j, k]
                            m += 1