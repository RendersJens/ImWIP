"""
:file:      operators_affine.py
:brief:     bla
:date:      20 DEC 2021
:author:    Jens Renders
            imec-Visionlab
            University of Antwerp
            jens.renders@uantwerpen.be
"""

import numpy as np
import imwip
try:
    import libimwip
    libimwip_available = True
except ImportError:
    libimwip_available = False


def affine_warp(
        f,
        A,
        b,
        f_warped=None,
        degree=3,
        indexing="ij",
        back_end=None
    ):

    if back_end is None:
        back_end = "cpp" if libimwip_available else "numba"

    dim = b.size
    if dim not in [2, 3]:
        raise ValueError("b should be of length 2 or 3")
    if back_end == "cpp":
        if dim == 2:
            warp_function = libimwip.affine_warp_2D
        else:
            warp_function = libimwip.affine_warp_3D
    elif back_end == "numba":
        if dim == 2:
            warp_function = imwip.numba_backend.affine_warp_2D
        else:
            warp_function = imwip.numba_backend.affine_warp_3D
    else:
        raise ValueError("back_end should be \"cpp\" or \"numba\"")

    return warp_function(
            f,
            A,
            b,
            f_warped,
            degree,
            indexing
        )


def adjoint_affine_warp(
        f_warped,
        A,
        b,
        f=None,
        degree=3,
        indexing="ij",
        back_end=None
    ):

    if back_end is None:
        back_end = "cpp" if libimwip_available else "numba"

    dim = b.size
    if dim not in [2, 3]:
        raise ValueError("b should be of length 2 or 3")
    if back_end == "cpp":
        if dim == 2:
            warp_function = libimwip.adjoint_affine_warp_2D
        else:
            warp_function = libimwip.adjoint_affine_warp_3D
    elif back_end == "numba":
        if dim == 2:
            warp_function = imwip.numba_backend.adjoint_affine_warp_2D
        else:
            warp_function = imwip.numba_backend.adjoint_affine_warp_3D
    else:
        raise ValueError("back_end should be \"cpp\" or \"numba\"")

    return warp_function(
            f_warped,
            A,
            b,
            f,
            degree,
            indexing
        )

def diff_affine_warp(
        f_warped,
        A,
        b,
        f=None,
        degree=3,
        indexing="ij",
        back_end=None
    ):

    if back_end is None:
        back_end = "cpp" if libimwip_available else "numba"

    dim = b.size
    if dim not in [2, 3]:
        raise ValueError("b should be of length 2 or 3")
    if back_end == "cpp":
        if dim == 2:
            warp_function = libimwip.adjoint_affine_warp_2D
        else:
            warp_function = libimwip.adjoint_affine_warp_3D
    elif back_end == "numba":
        if dim == 2:
            warp_function = imwip.numba_backend.adjoint_affine_warp_2D
        else:
            warp_function = imwip.numba_backend.adjoint_affine_warp_3D
    else:
        raise ValueError("back_end should be \"cpp\" or \"numba\"")

    return warp_function(
            f_warped,
            A,
            b,
            f,
            degree,
            indexing
        )