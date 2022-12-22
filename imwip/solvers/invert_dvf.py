"""
:file:      invert_dvf.py
:brief:     Iterative inversion of a DVF, which can be used
            for approximate adjoint or inverse warps.
:date:      20 DEC 2021
:author:    Jens Renders
            imec-Visionlab
            University of Antwerp
            jens.renders@uantwerpen.be
"""

import numpy as np
from tqdm import tqdm


def invert_dvf_2D(u, v, max_iter=15, verbose=False):
    inv_u = np.zeros(u.shape, dtype=np.float32)
    inv_v = np.zeros(v.shape, dtype=np.float32)

    if verbose:
        loop = tqdm(range(max_iter))
    else:
        loop = range(max_iter)

    for i in loop:
        inv_u = -warp_2D(u, inv_u, inv_v)
        inv_v = -warp_2D(v, inv_u, inv_v)

    return inv_u, inv_v


def invert_dvf_3D(u, v, w, max_iter=15, verbose=False):
    inv_u = np.zeros(u.shape, dtype=np.float32)
    inv_v = np.zeros(v.shape, dtype=np.float32)
    inv_w = np.zeros(w.shape, dtype=np.float32)

    if verbose:
        loop = tqdm(range(max_iter))
    else:
        loop = range(max_iter)

    for i in loop:
        inv_u = -warp_3D(u, inv_u, inv_v, inv_w)
        inv_v = -warp_3D(v, inv_u, inv_v, inv_w)
        inv_w = -warp_3D(w, inv_u, inv_v, inv_w)

    return inv_u, inv_v, inv_w