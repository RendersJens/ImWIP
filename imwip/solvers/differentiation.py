"""
:file:      differentiation.py
:brief:     provides differentiation functionality to
            ImWIP operators and pylops operators that contain ImWIP operators.
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
import pylops
import scipy.sparse as sps

def diff(A, x, to=None, matrix=False):
    """
    Given an imwip operator :math:`A = A(p)` (where :math:`p` represents all the parameters of
    :math:`A`), or a :py:mod:`pylops` block operator built of those, this function gives the derivative
    of :math:`A(p)x` towards :math:`p`.

    :param A: An ImWIP image warping operator or a Pylops block operator containing
        ImWIP image warping operators.
    :param x: A raveled image on which A acts
    :param to: a parameter or list of parameters to which to differentiate.
    :param matrix: if True, the derviative is returned as a :class:`scipy.sparse.coo_matrix`.
        Otherwise it will be returned as :class:`~scipy.sparse.linalg.LinearOperator`. Defaults to false

    :type A: :class:`~scipy.sparse.linalg.LinearOperator`
    :type x: :class:`numpy.ndarray`
    :type to: string or sequence of strings, optional
    :type matrix: bool, optional

    :return: The derivative or list of derivatives towards the parameters specified in `to`
    :rtype: :class:`~scipy.sparse.linalg.LinearOperator` or :class:`scipy.sparse.coo_matrix` or list of
        the same type.
    """
    if matrix:
        block_diag = sps.block_diag
    else:
        block_diag = pylops.BlockDiag

    if isinstance(A, pylops.Identity) or (hasattr(A, "constant") and A.constant):
        return [np.zeros((x.size, 0)) for var in to]
    elif hasattr(A, "derivative"):
        if isinstance(to, str):
            return A.derivative(x, to=[to])[0]
        else:
            return A.derivative(x, to=to)
    elif isinstance(A, pylops.VStack):
        if to is None:
            return block_diag([diff(Ai, x) for Ai in A.ops])
        elif isinstance(to, str):
            derivatives = []
            for Ai in A.ops:
                derivatives.append(diff(Ai, x, to=[to])[0])
            return block_diag(derivatives)
        else:
            diff_dict = {var: [] for var in to}
            for Ai in A.ops:
                derivatives = diff(Ai, x, to=to)
                for var, derivative in zip(to, derivatives):
                    diff_dict[var].append(derivative)
            return [block_diag(diff_dict[var]) for var in to]
    elif isinstance(A, pylops.BlockDiag):
        if to is None:
            raise NotImplementedError()
        elif isinstance(to, str):
            derivatives = []
            index = 0
            for Ai in A.ops:
                derivatives.append(diff(Ai, x[index:index + Ai.shape[1]], to=[to])[0])
                index += Ai.shape[1]
            return block_diag(derivatives)
        else:
            diff_dict = {var: [] for var in to}
            index = 0
            for Ai in A.ops:
                derivatives = diff(Ai, x[index:index + Ai.shape[1]], to=to)
                index += Ai.shape[1]
                for var, derivative in zip(to, derivatives):
                    diff_dict[var].append(derivative)
            return [block_diag(diff_dict[var]) for var in to]