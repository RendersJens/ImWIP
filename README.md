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
- cython

For conda users, these dependencies can be automatically installed from the `environment.yml` file in the root of this project. For example by running

```bash
conda env create
conda activate imwip
```

There is also a C++/CUDA backend, which is a bit faster than the numba backend, and it can
be accessed from any language that supports a C interface functions. It will automatically
compile on linux systems where the nvcc compiler is available.


Installation
------------

If git is installed, simply run

```bash
pip install git+https://github.com/RendersJens/ImWIP.git
```

Otherwise, download this repository and run pip in the root folder of the project:

```bash
pip install .
```

Getting started and reference documentation
-------------------------------------------
Full documentation is available on https://imwip.readthedocs.io/

Citing ImWIP
------------

If you use ImWIP in your research, please cite

Jens Renders, Ben Jeurissen, Anh-Tuan Nguyen, Jan De Beenhouwer, Jan Sijbers,
ImWIP: Open-source image warping toolbox with adjoints and derivatives,
SoftwareX,
Volume 24,
2023,
101524

BibTex:

```
@article{RENDERS2023101524,
  title = {ImWIP: Open-source image warping toolbox with adjoints and derivatives},
  author = {Jens Renders and Ben Jeurissen and Anh-Tuan Nguyen and Jan {De Beenhouwer} and Jan Sijbers},
  journal = {SoftwareX},
  volume = {24},
  pages = {101524},
  year = {2023},
  issn = {2352-7110},
  doi = {https://doi.org/10.1016/j.softx.2023.101524},
}
```
