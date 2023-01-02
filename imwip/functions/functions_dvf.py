"""
:file:      functions_dvf.py
:brief:     Image warping functions using a DVF
:author:    Jens Renders
"""

import numpy as np
import imwip.numba_backend
try:
    import libimwip
    libimwip_available = True
except ImportError:
    libimwip_available = False


def warp(
        image,
        u,
        v,
        w=None,
        out=None,
        degree=3,
        indexing="ij",
        backend=None
    ):
    """
    Warps a 2D or 3D function along a DVF.

    This function is linear in terms of the input f (even if degree 3
    is used for the splines). Therefore it has an adjoint function which is computed
    by :meth:`adjoint_warp`.

    :param image: Input image
    :param u: First component of the DVF describing the warp
    :param v: Second component of the DVF describing the warp
    :param w: Third component of the DVF describing the warp.
        Leave empty for a 2D warp
    :param out: Array to write the output image in.
        If None, an output array will be allocated.
    :param degree: Degree of the splines used for interpolation
    :param indexing: ``ij`` uses standard numpy array indexing. "xy" reversed the order of
        the indexes, making the vertical axis the first index. This can be more intuitive
        for 2D arrays. Defaults to ``ij``.
    :param backend: Whether to use the cpp or numba backend. If None, ``cpp`` will be used
        if available, else "numba"
    :type f: :class:`numpy.ndarray`
    :type u: :class:`numpy.ndarray`
    :type v: :class:`numpy.ndarray`
    :type w: :class:`numpy.ndarray`, optional
    :type out: :class:`numpy.ndarray`, optional
    :type degree: 1 or 3, optional
    :type indexing: ``ij`` or ``xy``, optional
    :type backend: ``cpp`` or ``numba``, optional
    :return: The warped image
    :rtype: :class:`numpy.ndarray`
    """

    if backend is None:
        backend = "cpp" if libimwip_available else "numba"

    dim = 2 if w is None else 3
    if backend == "cpp":
        if dim == 2:
            warp_function = libimwip.warp_2D
        else:
            warp_function = libimwip.warp_3D
    elif backend == "numba":
        if dim == 2:
            warp_function = imwip.numba_backend.warp_2D
        else:
            warp_function = imwip.numba_backend.warp_3D
    else:
        raise ValueError("backend should be \"cpp\" or \"numba\"")

    if dim == 2:
        return warp_function(
                f,
                u,
                v,
                f_warped,
                degree,
                indexing
            )
    else:
        return warp_function(
                f,
                u,
                v,
                w,
                f_warped,
                degree,
                indexing
            )

def adjoint_warp(
        image,
        u,
        v,
        w=None,
        out=None,
        degree=3,
        indexing="ij",
        backend=None
    ):
    """
    The function :meth:`warp` is a linear function of the input f (even if degree 3
    is used for the splines). Therefore it has an adjoint function which is computed
    by this function. See :meth:`warp` for the description of parameters and return value.

    """

    if backend is None:
        backend = "cpp" if libimwip_available else "numba"

    dim = 2 if w is None else 3
    if backend == "cpp":
        if dim == 2:
            warp_function = libimwip.adjoint_warp_2D
        else:
            warp_function = libimwip.adjoint_warp_3D
    elif backend == "numba":
        if dim == 2:
            warp_function = imwip.numba_backend.adjoint_warp_2D
        else:
            warp_function = imwip.numba_backend.adjoint_warp_3D
    else:
        raise ValueError("backend should be \"cpp\" or \"numba\"")

    if dim == 2:
        return warp_function(
                f_warped,
                u,
                v,
                f,
                degree,
                indexing
            )
    else:
        return warp_function(
                f_warped,
                u,
                v,
                w,
                f,
                degree,
                indexing
            )


def diff_warp(
        f_warped,
        u,
        v,
        w=None,
        diff_x=None,
        diff_y=None,
        diff_z=None,
        indexing="ij",
        backend=None
    ):
    """
    The derivative of :meth:`warp` towards the DVF. This function assumes splines of degree 3,
    to ensure differentiability.

    :param image: Input image
    :param u: First component of the DVF describing the warp
    :param v: Second component of the DVF describing the warp
    :param w: Third component of the DVF describing the warp.
        Leave empty for a 2D warp
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
        if available, else "numba"
    :type f: :class:`numpy.ndarray`
    :type u: :class:`numpy.ndarray`
    :type v: :class:`numpy.ndarray`
    :type w: :class:`numpy.ndarray`, optional
    :type diff_x: :class:`numpy.ndarray`, optional
    :type diff_y: :class:`numpy.ndarray`, optional
    :type diff_z: :class:`numpy.ndarray`, optional
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
            warp_function = libimwip.diff_warp_2D
        else:
            warp_function = libimwip.diff_warp_3D
    elif backend == "numba":
        if dim == 2:
            warp_function = imwip.numba_backend.diff_warp_2D
        else:
            warp_function = imwip.numba_backend.diff_warp_3D
    else:
        raise ValueError("backend should be \"cpp\" or \"numba\"")

    if dim == 2:
        return warp_function(
                f,
                u,
                v,
                diff_x,
                diff_y,
                indexing
            )
    else:
        return warp_function(
                f,
                u,
                v,
                w,
                diff_x,
                diff_y,
                diff_z,
                indexing
            )