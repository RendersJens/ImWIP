"""
:file:      wrappers.pyx
:brief:     Python wrappers of C/CUDA warping algorithms
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
cimport numpy as np
from libcpp.string cimport string

cdef extern from "utils.hu":
    int getDevice()
    void setDevice(int device)
    int getDeviceCount()
    string getDeviceName(int device)

def get_device():
    return getDevice()

def set_device(int device):
    setDevice(device)

def get_device_count():
    return getDeviceCount()

# def get_device_name(int device):
#     cdef string name = getDeviceName(device)
#     return name.decode('UTF-8')


# import the C++ versions of the warping functions
cdef extern from "warpAlgorithms.hu":
    void warp2D(
        const float* f,
        const float* u,
        const float* v,
        float* fWarped,
        int degree,
        int shape0,
        int shape1
    )

    void adjointWarp2D(
        const float* fWarped,
        const float* u,
        const float* v,
        float* f,
        int degree,
        int shape0,
        int shape1
    )

    void diffWarp2D(
        const float* f,
        const float* u,
        const float* v,
        float* diffx,
        float* diffy,
        int shape0,
        int shape1
    )

    void jvpWarp2D(
        const float* f,
        const float* u,
        const float* v,
        const float* vec_in,
        float* vec_out,
        int degree,
        int shape0,
        int shape1
    )

    void warp3D(
        const float* f,
        const float* u,
        const float* v,
        const float* w,
        float* fWarped,
        int degree,
        int shape0,
        int shape1,
        int shape2
    )

    void adjointWarp3D(
        const float* fWarped,
        const float* u,
        const float* v,
        const float* w,
        float* f,
        int degree,
        int shape0,
        int shape1,
        int shape2
    )

    void diffWarp3D(
        const float* f,
        const float* u,
        const float* v,
        const float* w,
        float* diffx,
        float* diffy,
        float* diffz,
        int shape0,
        int shape1,
        int shape2
    )

    void partialDiffWarp3D(
        const float* f,
        const float* u,
        const float* v,
        const float* w,
        int to,
        float* diff,
        int shape0,
        int shape1,
        int shape2
    )

    void jvpWarp3D(
        const float* f,
        const float* u,
        const float* v,
        const float* w,
        const float* input,
        float* output,
        int shape0,
        int shape1,
        int shape2
    )

    void jvpWarp3DY(
        const float* f,
        const float* u,
        const float* v,
        const float* w,
        const float* input,
        float* output,
        int shape0,
        int shape1,
        int shape2
    )

# python version of warp2D, this function accepts numpy arrays
def warp_2D(
        np.ndarray[ndim=2, dtype=float, mode="c"] f,
        np.ndarray[ndim=2, dtype=float, mode="c"] u,
        np.ndarray[ndim=2, dtype=float, mode="c"] v,
        np.ndarray[ndim=2, dtype=float, mode="c"] f_warped=None,
        int degree=3
    ):
    if f_warped is None:
        f_warped = np.zeros((f.shape[0], f.shape[1]), dtype=f.dtype)

    # the C++ version accepts pointers, so for each numpy array we
    # get the pointer to the first element and pass it to the C++ function
    warp2D(&f[0,0], &u[0,0], &v[0,0], &f_warped[0,0], degree, f.shape[0], f.shape[1])

    return f_warped


# python version of adjointWarp2D, this function accepts numpy arrays
def adjoint_warp_2D(
        np.ndarray[ndim=2, dtype=float, mode="c"] f_warped,
        np.ndarray[ndim=2, dtype=float, mode="c"] u,
        np.ndarray[ndim=2, dtype=float, mode="c"] v,
        np.ndarray[ndim=2, dtype=float, mode="c"] f=None,
        int degree=3
    ):
    if f is None:
        f = np.zeros((f_warped.shape[0], f_warped.shape[1]), dtype=f_warped.dtype)

    # the C++ version accepts pointers, so for each numpy array we
    # get the pointer to the first element and pass it to the C++ function
    adjointWarp2D(
        &f_warped[0,0],
        &u[0,0],
        &v[0,0],
        &f[0,0],
        degree,
        f.shape[0],
        f.shape[1]
    )

    return f


def diff_warp_2D(
        np.ndarray[ndim=2, dtype=float, mode="c"] f,
        np.ndarray[ndim=2, dtype=float, mode="c"] u,
        np.ndarray[ndim=2, dtype=float, mode="c"] v,
        np.ndarray[ndim=2, dtype=float, mode="c"] diff_x=None,
        np.ndarray[ndim=2, dtype=float, mode="c"] diff_y=None
    ):
    diff_x = np.zeros((f.shape[0], f.shape[1]), dtype=f.dtype)
    diff_y = np.zeros((f.shape[0], f.shape[1]), dtype=f.dtype)

    # the C++ version accepts pointers, so for each numpy array we
    # get the pointer to the first element and pass it to the C++ function
    diffWarp2D(&f[0,0], &u[0,0], &v[0, 0], &diff_x[0,0], &diff_y[0,0], f.shape[0], f.shape[1])

    return diff_x, diff_y


# python version of jvpWarp2D, this function accepts numpy arrays
def jvp_warp_2D(
        np.ndarray[ndim=2, dtype=float, mode="c"] f,
        np.ndarray[ndim=2, dtype=float, mode="c"] u,
        np.ndarray[ndim=2, dtype=float, mode="c"] v,
        np.ndarray[ndim=2, dtype=float, mode="c"] vec_in,
        np.ndarray[ndim=2, dtype=float, mode="c"] vec_out=None,
        int degree=3
    ):
    if vec_out is None:
        vec_out = np.zeros((2*f.shape[0], f.shape[1]), dtype=f.dtype)

    # the C++ version accepts pointers, so for each numpy array we
    # get the pointer to the first element and pass it to the C++ function
    jvpWarp2D(&f[0,0], &u[0,0], &v[0,0], &vec_in[0,0], &vec_out[0,0], degree, f.shape[0], f.shape[1])

    return vec_out


# python version of warp3D, this function accepts numpy arrays
def warp_3D(
        np.ndarray[ndim=3, dtype=float, mode="c"] f,
        np.ndarray[ndim=3, dtype=float, mode="c"] u,
        np.ndarray[ndim=3, dtype=float, mode="c"] v,
        np.ndarray[ndim=3, dtype=float, mode="c"] w,
        np.ndarray[ndim=3, dtype=float, mode="c"] f_warped=None,
        int degree=3
    ):
    if f_warped is None:
        f_warped = np.zeros((f.shape[0], f.shape[1], f.shape[2]), dtype=f.dtype)

    # the C++ version accepts pointers, so for each numpy array we
    # get the pointer to the first element and pass it to the C++ function
    warp3D(
        &f[0,0,0],
        &u[0,0,0],
        &v[0,0,0],
        &w[0,0,0],
        &f_warped[0,0,0],
        degree,
        f.shape[0],
        f.shape[1],
        f.shape[2]
    )

    return f_warped


# python version of adjointBackwardWarp3D, this function accepts numpy arrays
def adjoint_warp_3D(
        np.ndarray[ndim=3, dtype=float, mode="c"] f_warped,
        np.ndarray[ndim=3, dtype=float, mode="c"] u,
        np.ndarray[ndim=3, dtype=float, mode="c"] v,
        np.ndarray[ndim=3, dtype=float, mode="c"] w,
        np.ndarray[ndim=3, dtype=float, mode="c"] f=None,
        int degree=3
    ):
    if f is None:
        f = np.zeros((f_warped.shape[0], f_warped.shape[1], f_warped.shape[2]), dtype=f_warped.dtype)

    # the C++ version accepts pointers, so for each numpy array we
    # get the pointer to the first element and pass it to the C++ function
    adjointWarp3D(
        &f_warped[0,0,0],
        &u[0,0,0],
        &v[0,0,0],
        &w[0,0,0],
        &f[0,0,0],
        degree,
        f.shape[0],
        f.shape[1],
        f.shape[2]
    )

    return f


def diff_warp_3D(
        np.ndarray[ndim=3, dtype=float, mode="c"] f,
        np.ndarray[ndim=3, dtype=float, mode="c"] u,
        np.ndarray[ndim=3, dtype=float, mode="c"] v,
        np.ndarray[ndim=3, dtype=float, mode="c"] w,
        np.ndarray[ndim=3, dtype=float, mode="c"] diff_x=None,
        np.ndarray[ndim=3, dtype=float, mode="c"] diff_y=None,
        np.ndarray[ndim=3, dtype=float, mode="c"] diff_z=None,
    ):
    diff_x = np.zeros((f.shape[0], f.shape[1], f.shape[2]), dtype=f.dtype)
    diff_y = np.zeros((f.shape[0], f.shape[1], f.shape[2]), dtype=f.dtype)
    diff_z = np.zeros((f.shape[0], f.shape[1], f.shape[2]), dtype=f.dtype)

    # the C++ version accepts pointers, so for each numpy array we
    # get the pointer to the first element and pass it to the C++ function
    diffWarp3D(
        &f[0,0,0],
        &u[0,0,0],
        &v[0,0,0],
        &w[0,0,0],
        &diff_x[0,0,0],
        &diff_y[0,0,0],
        &diff_z[0,0,0],
        f.shape[0],
        f.shape[1],
        f.shape[2]
    )

    return diff_x, diff_y, diff_z


def partial_diff_warp_3D(
        np.ndarray[ndim=3, dtype=float, mode="c"] f,
        np.ndarray[ndim=3, dtype=float, mode="c"] u,
        np.ndarray[ndim=3, dtype=float, mode="c"] v,
        np.ndarray[ndim=3, dtype=float, mode="c"] w,
        int to,
        np.ndarray[ndim=3, dtype=float, mode="c"] diff=None
    ):
    diff = np.zeros((f.shape[0], f.shape[1], f.shape[2]), dtype=f.dtype)

    # the C++ version accepts pointers, so for each numpy array we
    # get the pointer to the first element and pass it to the C++ function
    partialDiffWarp3D(
        &f[0,0,0],
        &u[0,0,0],
        &v[0,0,0],
        &w[0,0,0],
        to,
        &diff[0,0,0],
        f.shape[0],
        f.shape[1],
        f.shape[2]
    )

    return diff


# python version of jvpWarp3D, this function accepts numpy arrays
def jvp_warp_3D(
        np.ndarray[ndim=3, dtype=float, mode="c"] f,
        np.ndarray[ndim=3, dtype=float, mode="c"] u,
        np.ndarray[ndim=3, dtype=float, mode="c"] v,
        np.ndarray[ndim=3, dtype=float, mode="c"] w,
        np.ndarray[ndim=3, dtype=float, mode="c"] vec_in,
        np.ndarray[ndim=3, dtype=float, mode="c"] vec_out=None
    ):
    if vec_out is None:
        vec_out = np.zeros((3*f.shape[0], f.shape[1], f.shape[2]), dtype=f.dtype)

    # the C++ version accepts pointers, so for each numpy array we
    # get the pointer to the first element and pass it to the C++ function
    jvpWarp3D(
        &f[0,0,0],
        &u[0,0,0],
        &v[0,0,0],
        &w[0,0,0],
        &vec_in[0,0,0],
        &vec_out[0,0,0],
        f.shape[0],
        f.shape[1],
        f.shape[2]
    )

    return vec_out

# python version of diffWarp3DY, this function accepts numpy arrays
def jvp_warp_3DY(
        np.ndarray[ndim=3, dtype=float, mode="c"] f,
        np.ndarray[ndim=3, dtype=float, mode="c"] u,
        np.ndarray[ndim=3, dtype=float, mode="c"] v,
        np.ndarray[ndim=3, dtype=float, mode="c"] w,
        np.ndarray[ndim=3, dtype=float, mode="c"] vec_in,
        np.ndarray[ndim=3, dtype=float, mode="c"] vec_out=None
    ):
    if vec_out is None:
        vec_out = np.zeros((f.shape[0], f.shape[1], f.shape[2]), dtype=f.dtype)

    # the C++ version accepts pointers, so for each numpy array we
    # get the pointer to the first element and pass it to the C++ function
    jvpWarp3DY(
        &f[0,0,0],
        &u[0,0,0],
        &v[0,0,0],
        &w[0,0,0],
        &vec_in[0,0,0],
        &vec_out[0,0,0],
        f.shape[0],
        f.shape[1],
        f.shape[2]
    )

    return vec_out


# import the C++ versions of the affine warping functions
cdef extern from "warpAlgorithmsAffine.hu":
    void affineWarp2D(
        const float* f,
        const float* A,
        const float* b,
        float* fWarped,
        int degree,
        int shape0,
        int shape1
    )

    void adjointAffineWarp2D(
        const float* fWarped,
        const float* A,
        const float* b,
        float* f,
        int degree,
        int shape0,
        int shape1
    )

    void diffAffineWarp2D(
        const float* f,
        const float* A,
        const float* b,
        float* diffx,
        float* diffy,
        int shape0,
        int shape1
    )

    void affineWarp3D(
        const float* f,
        const float* A,
        const float* b,
        float* fWarped,
        int degree,
        int shape0,
        int shape1,
        int shape2
    )

    void diffAffineWarp3D(
        const float* f,
        const float* A,
        const float* b,
        float* diffx,
        float* diffy,
        float* diffz,
        int shape0,
        int shape1,
        int shape2
    )

    void adjointAffineWarp3D(
        const float* fWarped,
        const float* A,
        const float* b,
        float* f,
        int degree,
        int shape0,
        int shape1,
        int shape2
    )


# python version of affineWarp2D, this function accepts numpy arrays
def affine_warp_2D(
        np.ndarray[ndim=2, dtype=float, mode="c"] f,
        np.ndarray[ndim=2, dtype=float, mode="c"] A,
        np.ndarray[ndim=1, dtype=float, mode="c"] b,
        np.ndarray[ndim=2, dtype=float, mode="c"] f_warped=None,
        int degree=3,
        str indexing="ij"
    ):
    if f_warped is None:
        f_warped = np.zeros((f.shape[0], f.shape[1]), dtype=f.dtype)

    if indexing == "xy":
        A = np.fliplr(np.flipud(A)).copy()
        b = np.flip(b).copy()

    # the C++ version accepts pointers, so for each numpy array we
    # get the pointer to the first element and pass it to the C++ function
    affineWarp2D(&f[0,0], &A[0,0], &b[0], &f_warped[0,0], degree, f.shape[0], f.shape[1])

    return f_warped


# python version of adjointAffineWarp2D, this function accepts numpy arrays
def adjoint_affine_warp_2D(
        np.ndarray[ndim=2, dtype=float, mode="c"] f_warped,
        np.ndarray[ndim=2, dtype=float, mode="c"] A,
        np.ndarray[ndim=1, dtype=float, mode="c"] b,
        np.ndarray[ndim=2, dtype=float, mode="c"] f=None,
        int degree=3,
        str indexing = "ij"
    ):
    if f is None:
        f = np.zeros((f_warped.shape[0], f_warped.shape[1]), dtype=f_warped.dtype)

    if indexing == "xy":
        A = np.fliplr(np.flipud(A)).copy()
        b = np.flip(b).copy()

    # the C++ version accepts pointers, so for each numpy array we
    # get the pointer to the first element and pass it to the C++ function
    adjointAffineWarp2D(
        &f_warped[0,0],
        &A[0,0],
        &b[0],
        &f[0,0],
        degree,
        f.shape[0],
        f.shape[1]
    )

    return f

def diff_affine_warp_2D(
        np.ndarray[ndim=2, dtype=float, mode="c"] f,
        np.ndarray[ndim=2, dtype=float, mode="c"] A,
        np.ndarray[ndim=1, dtype=float, mode="c"] b,
        np.ndarray[ndim=2, dtype=float, mode="c"] diff_x=None,
        np.ndarray[ndim=2, dtype=float, mode="c"] diff_y=None,
        str indexing="ij"
    ):
    diff_x = np.zeros((f.shape[0], f.shape[1]), dtype=f.dtype)
    diff_y = np.zeros((f.shape[0], f.shape[1]), dtype=f.dtype)

    if indexing == "xy":
        A = np.fliplr(np.flipud(A)).copy()
        b = np.flip(b).copy()

    # the C++ version accepts pointers, so for each numpy array we
    # get the pointer to the first element and pass it to the C++ function
    if indexing == "xy":
        diffAffineWarp2D(&f[0,0], &A[0,0], &b[0], &diff_y[0,0], &diff_x[0,0], f.shape[0], f.shape[1])
    else:
        diffAffineWarp2D(&f[0,0], &A[0,0], &b[0], &diff_x[0,0], &diff_y[0,0], f.shape[0], f.shape[1])

    return diff_x, diff_y

# python version of affineWarp3D, this function accepts numpy arrays
def affine_warp_3D(
        np.ndarray[ndim=3, dtype=float, mode="c"] f,
        np.ndarray[ndim=2, dtype=float, mode="c"] A,
        np.ndarray[ndim=1, dtype=float, mode="c"] b,
        np.ndarray[ndim=3, dtype=float, mode="c"] f_warped=None,
        int degree=3,
        str indexing="ij"
    ):
    if f_warped is None:
        f_warped = np.zeros((f.shape[0], f.shape[1], f.shape[2]), dtype=f.dtype)

    if indexing == "xy":
        A = np.fliplr(np.flipud(A)).copy()
        b = np.flip(b).copy()

    # the C++ version accepts pointers, so for each numpy array we
    # get the pointer to the first element and pass it to the C++ function
    affineWarp3D(
        &f[0,0,0],
        &A[0,0],
        &b[0],
        &f_warped[0,0,0],
        degree,
        f.shape[0],
        f.shape[1],
        f.shape[2]
    )

    return f_warped

# python version of adjointAffineWarp3D, this function accepts numpy arrays
def adjoint_affine_warp_3D(
        np.ndarray[ndim=3, dtype=float, mode="c"] f_warped,
        np.ndarray[ndim=2, dtype=float, mode="c"] A,
        np.ndarray[ndim=1, dtype=float, mode="c"] b,
        np.ndarray[ndim=3, dtype=float, mode="c"] f=None,
        int degree=3,
        str indexing="ij"
    ):
    if f is None:
        f = np.zeros((f_warped.shape[0], f_warped.shape[1], f_warped.shape[2]), dtype=f_warped.dtype)

    if indexing == "xy":
        A = np.fliplr(np.flipud(A)).copy()
        b = np.flip(b).copy()

    # the C++ version accepts pointers, so for each numpy array we
    # get the pointer to the first element and pass it to the C++ function
    adjointAffineWarp3D(
        &f_warped[0,0,0],
        &A[0,0],
        &b[0],
        &f[0,0,0],
        degree,
        f.shape[0],
        f.shape[1],
        f.shape[2]
    )

    return f

def diff_affine_warp_3D(
        np.ndarray[ndim=3, dtype=float, mode="c"] f,
        np.ndarray[ndim=2, dtype=float, mode="c"] A,
        np.ndarray[ndim=1, dtype=float, mode="c"] b,
        np.ndarray[ndim=3, dtype=float, mode="c"] diff_x=None,
        np.ndarray[ndim=3, dtype=float, mode="c"] diff_y=None,
        np.ndarray[ndim=3, dtype=float, mode="c"] diff_z=None,
        str indexing="ij"
    ):
    
    diff_x = np.zeros((f.shape[0], f.shape[1], f.shape[2]), dtype=f.dtype)
    diff_y = np.zeros((f.shape[0], f.shape[1], f.shape[2]), dtype=f.dtype)
    diff_z = np.zeros((f.shape[0], f.shape[1], f.shape[2]), dtype=f.dtype)

    if indexing == "xy":
        A = np.fliplr(np.flipud(A)).copy()
        b = np.flip(b).copy()

    # the C++ version accepts pointers, so for each numpy array we
    # get the pointer to the first element and pass it to the C++ function
    diffAffineWarp3D(
        &f[0,0,0],
        &A[0,0],
        &b[0],
        &diff_x[0,0,0],
        &diff_y[0,0,0],
        &diff_z[0,0,0],
        f.shape[0],
        f.shape[1],
        f.shape[2]
    )

    return diff_x, diff_y, diff_z