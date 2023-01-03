"""
:file:      invert_dvf.py
:brief:     Iterative inversion of a DVF, which can be used
            for approximate adjoint or inverse warps.
:author:    Jens Renders
"""

import numpy as np
from tqdm import tqdm


def invert_dvf_2D(u, v, max_iter=15, verbose=False, backend=None):
    """
    Approximately inverts a 2D DVF using the fixed point inversion technique
    of :cite:t:`chen2008simple`

    .. note::
        A DVF can usually not be inverted exactly, since the vectors are only defined
        on integer coordinates while they point at real coordinates.

    :param u: First component of the DVF describing the warp
    :param v: Second component of the DVF describing the warp
    :param max_iter: Number of fixed point iterations to perform, defaults to 15.
    :param verbose: whether to show a progress bar, defaults to False.
    :param backend: Whether to use the cpp or numba backend. If None, ``cpp`` will be used
        if available, else ``numba``

    :type u: :class:`numpy.ndarray`
    :type v: :class:`numpy.ndarray`
    :type max_iter: int, optional
    :type verbose: bool, optional
    :type backend: ``cpp`` or ``numba``, optional

    :return: inv_u, inv_v: the two components of the inverted dvf
    :rtype: :class:`numpy.ndarray`, :class:`numpy.ndarray`
    """
    inv_u = np.zeros(u.shape, dtype=np.float32)
    inv_v = np.zeros(v.shape, dtype=np.float32)

    if verbose:
        loop = tqdm(range(max_iter))
    else:
        loop = range(max_iter)

    for i in loop:
        inv_u = -imwip.warp(u, inv_u, inv_v, backend=backend)
        inv_v = -imwip.warp(v, inv_u, inv_v, backend=backend)

    return inv_u, inv_v


def invert_dvf_3D(u, v, w, max_iter=15, verbose=False, backend=None):
    """
    Approximately inverts a 3D DVF using the fixed point inversion technique
    of :cite:t:`chen2008simple`

    .. note::
        A DVF can usually not be inverted exactly, since the vectors are only defined
        on integer coordinates while they point at real coordinates.

    :param u: First component of the DVF describing the warp
    :param v: Second component of the DVF describing the warp
    :param w: Second component of the DVF describing the warp
    :param max_iter: Number of fixed point iterations to perform. Defaults to 15.
    :param verbose: whether to show a progress bar. Defaults to False.
    :param backend: Whether to use the cpp or numba backend. If None, ``cpp`` will be used
        if available, else ``numba``

    :type u: :class:`numpy.ndarray`
    :type v: :class:`numpy.ndarray`
    :type w: :class:`numpy.ndarray`
    :type max_iter: int, optional
    :type verbose: bool, optional
    :type backend: ``cpp`` or ``numba``, optional

    :return: inv_u, inv_v, inv_w: the three components of the inverted dvf
    :rtype: :class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`numpy.ndarray`
    """
    inv_u = np.zeros(u.shape, dtype=np.float32)
    inv_v = np.zeros(v.shape, dtype=np.float32)
    inv_w = np.zeros(w.shape, dtype=np.float32)

    if verbose:
        loop = tqdm(range(max_iter))
    else:
        loop = range(max_iter)

    for i in loop:
        inv_u = -imwip.warp(u, inv_u, inv_v, inv_w, backend=backend)
        inv_v = -imwip.warp(v, inv_u, inv_v, inv_w, backend=backend)
        inv_w = -imwip.warp(w, inv_u, inv_v, inv_w, backend=backend)

    return inv_u, inv_v, inv_w