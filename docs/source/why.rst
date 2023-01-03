Why ImWIP?
==========

ImWIP is short for **Image Warping** for **Inverse Problems**

Image warping is a powerful tool for modeling motion and deformation of 2D and 3D images. For example, it can provide a model that simulates dynamic CT scans, or MRI scans with patient motion. There are many excellent packages that provide image warping functionality. For example

* sckikit-image
    - :py:func:`skimage.transform.warp` (2D affine and ND DVF)
* SciPy
    - :py:func:`scipy.ndimage.map_coordinates` (ND DVF)
    - :py:func:`scipy.ndimage.affine_transform` (ND affine)
* OpenCV
    - :py:func:`cv2.remap` (2D DVF)

The equivalent functions in ImWIP are

* :py:func:`~imwip.functions.functions_dvf.warp` (2D and 3D DVF)
* :py:func:`~imwip.functions.functions_affine.affine_warp` (2D and 3D affine)

These packages allow the user to easily implement such models, but they lack some important
tools for the **inversion** of such models.

The matrix representation of image warping
------------------------------------------

The warping functions listed above are all linear maps in terms of the input image. This
means that it can be represented by a matrix, that acts (by matrix-vector multiplication)
in an image, represented as a raveled vector.

In theory, this matrix can be extracted from those functions, by applying the functions
to each column of a unit matrix. In other words, by warping an image with one pixel set to
1, while all other pixels are set to 0, we obtain one column of the image warping matrix.
This is very expensive, as it requires one warp per pixel.

With ImWIP, the matrix corresponding to an image warp can be constructed efficiently, using
:py:mod:`imwip.matrices`. The computational cost is about the same as a single image warp, and
the memory cost is reduced by constructing a sparse matrix.

Using this matrix, a lot of inverse problems can already be solved using linear algebra and matrix-vector calculus. However, even when using sparse matrices, the memory cost can be quite high. Especially when working with large 3D images. A solution is given by using Adjoint image warping and
:py:class:`~scipy.sparse.linalg.LinearOperator` 

Adjoint image warping and operators
-----------------------------------

Many techniques in linear algebra and optimization do not require explicit matrices. They only need to know how the matrix acts on a vector (which is the image warping function itself), and how the transpose of this matrix acts on a vector. This is adjoint image warping. It is implemented as a function in

* :py:func:`~imwip.functions.functions_dvf.adjoint_warp`
* :py:func:`~imwip.functions.functions_affine.adjoint_affine_warp`

And for convenience, the regular and adjoint warps are combined together in a :py:class:`~scipy.sparse.linalg.LinearOperator`, which can be used as if it is a matrix, without the memory costs. These operators can be found in :py:mod:`imwip.operators`

Differentiated image warping
----------------------------

The linear algebra discussed above is usefull for inverse problems where the image (which deforms or moves) is unknown. In the case that the motion or deformation itself is unknown, we are dealing with a non-linear problems, which can often be solved efficiently with gradient based techniques.

In this case, the derivatives of image warping towards the DVF or othe motion parameters is required. They are implemented in

* :py:func:`~imwip.functions.functions_dvf.diff_warp`
* :py:func:`~imwip.functions.functions_affine.diff_affine_warp`

and the :py:class:`~scipy.sparse.linalg.LinearOperator` s can be differentiated using :py:func:`~imwip.solvers.differentiation.diff`
