imwip.operators
===============
.. automodule:: imwip.operators
   :members:
   :undoc-members:
   :show-inheritance:

Image warping is linear in terms of the input image (even if degree 3 is used for the
splines). This means that the warp can be represented by a matrix :code:`M`, such that for a
(raveled) image :code:`x`, the warped image is given by :code:`M @ x`, where :code:`@` is Python's
matmul (matrix multiplication) operator. The adjoint warp is given by :code:`M.T @ x` or
:code:`x @ M`. The matrix :code:`M` can be obtained explicitly, using :py:mod:`imwip.matrices`.

This module provides Scipy LinearOperators which are equivalent to the matrix representation,
but avoid the memory use of explicly storing the matrix. It replaces the matrix-vecor
multiplication with the efficient GPU implementations of warps and adjoint warps. This allows for concise and expressive notation of many mathematical algorithms without any performance cost. It is compatible with other packages that use SciPy LinearOperators such as Pylops
and the solvers of :mod:`scipy.sparse.linalg`.

imwip.operators.operators\_dvf
------------------------------

.. automodule:: imwip.operators.operators_dvf
   :members:
   :undoc-members:
   :show-inheritance:

imwip.operators.operators\_affine
---------------------------------

.. automodule:: imwip.operators.operators_affine
   :members:
   :undoc-members:
   :show-inheritance: