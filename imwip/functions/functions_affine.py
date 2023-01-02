"""
:file:      functions_affine.py
:brief:     Image warping functions using an affine transformation
:author:    Jens Renders
"""

import numpy as np
import imwip.numba_backend
try:
    import libimwip
    libimwip_available = True
except ImportError:
    libimwip_available = False


def affine_warp(
        image,
        A,
        b,
        out=None,
        degree=3,
        indexing="ij",
        backend=None
    ):
    """
    Warps a 2D or 3D image according to an affine transformation Ax + b.

    This function is linear in terms of the input image (even if degree 3
    is used for the splines). Therefore it has an adjoint function which is computed
    by :meth:`adjoint_affine_warp`.

    :param image: Input image
    :param A: Matrix A of the affine transformation Ax + b.
        It shoud be an 2x2 for 2D warp and 3x3 for a 3D warp.
    :param b: Vector b of the affine transformation Ax + b. It should have length 2 for
        a 2D warp and length 3 for a 3D warp.
    :param out: Array to write the output image in.
        If None, an output array will be allocated.
    :param degree: Degree of the splines used for interpolation
    :param indexing: ``ij`` uses standard numpy array indexing. "xy" reversed the order of
        the indexes, making the vertical axis the first index. This can be more intuitive
        for 2D arrays. Defaults to ``ij``.
    :param backend: Whether to use the cpp or numba backend. If None, ``cpp`` will be used
        if available, else ``numba``
    :type image: :class:`numpy.ndarray`
    :type A: :class:`numpy.ndarray`
    :type b: :class:`numpy.ndarray`
    :type out: :class:`numpy.ndarray`, optional
    :type degree: 1 or 3, optional
    :type indexing: ``ij`` or ``xy``, optional
    :type backend: ``cpp`` or ``numba``, optional
    :return: The warped image
    :rtype: :class:`numpy.ndarray`
    """

    if backend is None:
        backend = "cpp" if libimwip_available else "numba"

    dim = b.size
    if dim not in [2, 3]:
        raise ValueError("b should be of length 2 or 3")
    if backend == "cpp":
        if dim == 2:
            warp_function = libimwip.affine_warp_2D
        else:
            warp_function = libimwip.affine_warp_3D
    elif backend == "numba":
        if dim == 2:
            warp_function = imwip.numba_backend.affine_warp_2D
        else:
            warp_function = imwip.numba_backend.affine_warp_3D
    else:
        raise ValueError("backend should be \"cpp\" or \"numba\"")

    return warp_function(
            image,
            A,
            b,
            out,
            degree,
            indexing
        )


def adjoint_affine_warp(
        image,
        A,
        b,
        out=None,
        degree=3,
        indexing="ij",
        backend=None
    ):
    """
    The function :meth:`affine_warp` is a linear function of the input image (even if degree 3
    is used for the splines). Therefore it has an adjoint function which is computed
    by this function. See :meth:`affine_warp` for the description of parameters and return value.

    """

    if backend is None:
        backend = "cpp" if libimwip_available else "numba"

    dim = b.size
    if dim not in [2, 3]:
        raise ValueError("b should be of length 2 or 3")
    if backend == "cpp":
        if dim == 2:
            warp_function = libimwip.adjoint_affine_warp_2D
        else:
            warp_function = libimwip.adjoint_affine_warp_3D
    elif backend == "numba":
        if dim == 2:
            warp_function = imwip.numba_backend.adjoint_affine_warp_2D
        else:
            warp_function = imwip.numba_backend.adjoint_affine_warp_3D
    else:
        raise ValueError("backend should be \"cpp\" or \"numba\"")

    return warp_function(
            image,
            A,
            b,
            out,
            degree,
            indexing
        )


def diff_affine_warp(
        image,
        A,
        b,
        diff_x=None,
        diff_y=None,
        diff_z=None,
        indexing="ij",
        backend=None
    ):
    """
    The derivative of :meth:`affine_warp` towards the DVF describing the affine warp.
    This function assumes splines of degree 3, to ensure differentiability.

    :param image: Input image
    :param A: Matrix A of the affine transformation Ax + b.
        It shoud be an 2x2 for 2D warp and 3x3 for a 3D warp.
    :param b: Vector b of the affine transformation Ax + b. It should have length 2 for
        a 2D warp and length 3 for a 3D warp.
    :param diff_x: Array to write the derivative to the first component in.
        If None, an output array will be allocated.
    :param diff_y: Array to write the derivative to the first component in.
        If None, an output array will be allocated.
    :param diff_z: Array to write the derivative to the first component in.
        If None, an output array will be allocated.
    :param indexing: ``ij`` uses standard numpy array indexing. "xy" reversed the order of
        the indexes, making the vertical axis the first index. This can be more intuitive
        for 2D arrays. Defaults to ``ij``.
    :param backend: Whether to use the cpp or numba backend. If None, ``cpp`` will be used
        if available, else ``numba``
    :type image: :class:`numpy.ndarray`
    :type u: :class:`numpy.ndarray`
    :type v: :class:`numpy.ndarray`
    :type w: :class:`numpy.ndarray`, optional
    :type A: :class:`numpy.ndarray`
    :type b: :class:`numpy.ndarray`
    :type indexing: ``ij`` or ``xy``, optional
    :type backend: ``cpp`` or ``numba``, optional
    :return: diff_x, diff_y, diff_z
    :rtype: :class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`numpy.ndarray`, 
    """

    if backend is None:
        backend = "cpp" if libimwip_available else "numba"

    dim = b.size
    if dim not in [2, 3]:
        raise ValueError("b should be of length 2 or 3")
    if backend == "cpp":
        if dim == 2:
            warp_function = libimwip.diff_affine_warp_2D
        else:
            warp_function = libimwip.diff_affine_warp_3D
    elif backend == "numba":
        if dim == 2:
            warp_function = imwip.numba_backend.diff_affine_warp_2D
        else:
            warp_function = imwip.numba_backend.diff_affine_warp_3D
    else:
        raise ValueError("backend should be \"cpp\" or \"numba\"")

    if dim == 2:
        return warp_function(
                image,
                A,
                b,
                diff_x,
                diff_y,
                indexing
            )
    else:
        return warp_function(
                image,
                A,
                b,
                diff_x,
                diff_y,
                diff_z,
                indexing
            )