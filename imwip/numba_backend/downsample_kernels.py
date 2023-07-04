"""
:file:      downsample_kernels.py
:brief:     Downsample kernels using numba
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
def cubic_downsampling_2D_kernel(f, f_lr, coeffs, adjoint):
    i, j = cuda.grid(2)

    if i < f_lr.shape[0] and j < f_lr.shape[1]:
        f_i = float32(i)
        f_j = float32(j)

        F_i = float32(f.shape[0]/f_lr.shape[0])
        F_j = float32(f.shape[1]/f_lr.shape[1])

        # position at which to iterpolate
        x = f_i * F_i + (F_i - float32(1))/float32(2)
        y = f_j * F_j + (F_j - float32(1))/float32(2)

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
                        cuda.atomic.add(f, (Q0, Q1), coefficient * f_lr[i, j])
                    else:
                        f_lr[i, j] += coefficient * f[Q0, Q1]
                m += 1


@cuda.jit
def cubic_downsampling_3D_kernel(f, f_lr, coeffs, adjoint):
    i, j, k = cuda.grid(3)

    if i < f_lr.shape[0] and j < f_lr.shape[1] and k < f_lr.shape[2]:
        f_i = float32(i)
        f_j = float32(j)
        f_k = float32(k)

        F_i = float32(f.shape[0]/f_lr.shape[0])
        F_j = float32(f.shape[1]/f_lr.shape[1])
        F_k = float32(f.shape[2]/f_lr.shape[2])

        # position at which to iterpolate
        x = f_i * F_i + (F_i - float32(1))/float32(2)
        y = f_j * F_j + (F_j - float32(1))/float32(2)
        z = f_k * F_k + (F_k - float32(1))/float32(2)

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
                            cuda.atomic.add(f, (Q0, Q1, Q2), coefficient * f_lr[i, j, k])
                        else:
                            f_lr[i, j, k] += coefficient * f[Q0, Q1, Q2]
                    m += 1
