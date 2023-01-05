[![Documentation Status](https://readthedocs.org/projects/imwip/badge/?version=latest)](https://imwip.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/452688446.svg)](https://zenodo.org/badge/latestdoi/452688446)


**ImWIP**: Image Warping for Inverse Problems

ImWIP provides efficient, matrix-free and GPU accelerated implementations of image warping operators, in Python and C++. The goal of this package is to enable the use of image warping in inverse problems. This requires two extra operations on top of regular image warping: adjoint image warping (to solve for images) and differentiated image warping (to solve for the deformation field).


Requirements
------------

ImWIP heavily relies on CUDA kernels for efficient parallelization. Therfore a CUDA enabeled GPU is required. Furthermore, the following python dependencies are needed, which can be easily installed using conda, and get the package working using the numba/CUDA backend.

- python >= 3.8
- numpy
- scipy
- numba
- pylops
- tqdm

There is also a C++/CUDA backend, which is a bit faster then the numba backend, and it can
be accessed from any language that supports a C interface functions. It will automatically
compile if the following dependencies are met:

- linux
- cython
- nvcc

Installation
------------

If git is installed, simply run

`$ pip install git+https://github.com/RendersJens/ImWIP.git`


Otherwise, download this repository and run pip in the root folder of the project:

`$ pip install .`

Getting started and reference documentation
-------------------------------------------
Full documentation is available on https://imwip.readthedocs.io/
