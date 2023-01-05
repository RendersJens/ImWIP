"""
:file:      warp_kernels.py
:brief:     Warping kernels using numba
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
from numba import cuda
from numba.types import void, int32, float32, boolean
import math


@cuda.jit
def linear_warp_2D_kernel(f, u, v, f_warped, adjoint):
    i, j = cuda.grid(2)

    if i < f.shape[0] and j < f.shape[1]:
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
            if 0 <= Q[m][0] < f.shape[0] and 0 <= Q[m][1] < f.shape[1]:
                if adjoint:
                    cuda.atomic.add(f_warped, (Q[m][0], Q[m][1]), coefficients[m] * f[i, j])
                else:
                    f_warped[i, j] += coefficients[m] * f[Q[m][0], Q[m][1]]


@cuda.jit
def linear_warp_3D_kernel(f, u, v, w, f_warped, adjoint):
    i, j, k = cuda.grid(3)

    if i < f.shape[0] and j < f.shape[1] and k < f.shape[2]:
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
            if 0 <= Q[m][0] < f.shape[0] and 0 <= Q[m][1] < f.shape[1] and 0 <= Q[m][2] < f.shape[2]:
                if adjoint:
                    cuda.atomic.add(f_warped, (Q[m][0], Q[m][1], Q[m][2]), coefficients[m] * f[i, j, k])
                else:
                    f_warped[i, j, k] += coefficients[m] * f[Q[m][0], Q[m][1], Q[m][2]]


@cuda.jit
def cubic_warp_2D_kernel(f, u, v, f_warped, coeffs, adjoint):
    i, j = cuda.grid(2)

    if i < f.shape[0] and j < f.shape[1]:
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
                if 0 <= Q0 < f.shape[0] and 0 <= Q1 < f.shape[1]:
                    coefficient = float32(0)
                    for n in range(16):
                        coefficient += coeffs[m*16 + n] * monomials[n]
                    if adjoint:
                        cuda.atomic.add(f_warped, (Q0, Q1), coefficient * f[i, j])
                    else:
                        f_warped[i, j] += coefficient * f[Q0, Q1]
                m += 1


@cuda.jit
def cubic_warp_3D_kernel(f, u, v, w, f_warped, coeffs, adjoint):
    i, j, k = cuda.grid(3)

    if i < f.shape[0] and j < f.shape[1] and k < f.shape[2]:
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
                    if 0 <= Q0 < f.shape[0] and 0 <= Q1 < f.shape[1] and 0 <= Q2 < f.shape[2]:
                        coefficient = float32(0)
                        for n in range(64):
                            coefficient += coeffs[m*64 + n] * monomials[n]
                        if adjoint:
                            cuda.atomic.add(f_warped, (Q0, Q1, Q2), coefficient * f[i, j, k])
                        else:
                            f_warped[i, j, k] += coefficient * f[Q0, Q1, Q2]
                    m += 1