"""
:file:      invert_dvf.py
:brief:     Iterative inversion of a DVF, which can be used
            for approximate adjoint or inverse warps.
:author:    Jens Renders
"""

import numpy as np
from tqdm import tqdm


def invert_dvf_2D(u, v, max_iter=15, verbose=False, backend=None):
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