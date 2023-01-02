imwip.matrices
===============
.. automodule:: imwip.matrices
   :members:
   :undoc-members:
   :show-inheritance:

Image warping is linear in terms of the input image (even if degree 3 is used for the
splines). This means that the warp can be represented by a matrix :code:`M`, such that for a
(raveled) image :code:`x`, the warped image is given by :code:`M @ x`, where :code:`@` is Python's
matmul (matrix multiplication) operator. The adjoint warp is given by :code:`M.T @ x` or
:code:`x @ M`. This module provides functions that construct this matrix as a
:class:`scipy.sparse.coo_matrix` (which can then be converted to any other matrix type).

.. note::
    :py:mod:`imwip.operators` provides Scipy LinearOperators which are equivalent
    to the matrix representation, but avoid the memory use of explicly storing the matrix.
    It replaces the matrix-vector multiplication with the efficient GPU implementations of
    warps and adjoint warps. If you only need matrix-vector multiplications, this will be
    more efficient. Only use the matrices if you need explicit access to its coefficients.

imwip.matrices.matrices\_dvf
------------------------------

.. automodule:: imwip.matrices.matrices_dvf
   :members:
   :undoc-members:
   :show-inheritance:


imwip.matrices.matrices\_affine
---------------------------------

.. automodule:: imwip.matrices.matrices_affine
   :members:
   :undoc-members:
   :show-inheritance: