[![DOI](https://zenodo.org/badge/452688446.svg)](https://zenodo.org/badge/latestdoi/452688446)


**ImWIP**: CUDA/C implementations of various warping and adjoint warping and differentiated warping algorithms, with python wrappers.

Features
------------

* Linear and cubic image warping of 2D and 3D images
  * Using a Deformation Vector Field (DVF)
  * Using an affine transformation
* The adjoint action of the above warp functions. Each of these image warps can be seen as a linear operator $A$ acting on a vector $x$ that represents the image. The implementation of the adjoint operators $A^T$ or $A^*$ is usefull to solve linear systems involving $A$ and to compute analytic derivatives to $x$ of functionals involving $Ax$.
* Analytic derivatives of $A(t)$ to $t$, where $A(t)$ is a warping operator along rigid or affine motion determined by the vector $t$ of rigid or affine motion parameters. This is a basic tool in the development of algorithms that solve for the motion parameters.

As an example, imagine that we want to solve the following system for $t$ and $x$:
$$
BA(t)x = b
$$

or similarly, we want to minimize
$$
f(x,t) = \frac{1}{2}\lVert BA(t)x - b \rVert_2^2
$$

Here $b$ can represent data that is the result of moving an unknown image $x$ with unknown affine motion and then applying a known linear transformation $B$. To solve this problem, we need the gradient of $f$ with respect to $x$ and $t$:
$$
\begin{aligned}
\nabla_x f(x,t) &= &A^T(t) &B^T(BA(t) x - b)\\
\nabla_t f(x,t) &= &(A'(t)x)^T &B^T(BA(t) x - b)
\end{aligned}
$$
This requires the operators $A(t)$, $A^T(t)$ and $A'(t)$, which are all provided by this package.

Requirements
------------
* gcc and g++
* nvcc
* Python 3.7+
    * numpy
    * scipy
    * numba
    * pylops (currently requires scipy < 1.8)

Installation
------------
**Install with pip:**

`$ pip install git+https://github.com/RendersJens/ImWIP.git`


Or clone/download the repository and run pip in the root folder of the project:

`$ pip install .`

**Install without pip:**

clone/download the repository and run

`$ python setup.py install`

in the root folder of the project.

Basic usage
-----------
Take a look at `/examples` for basic usage.
