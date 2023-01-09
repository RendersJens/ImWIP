Installation
============

Requirements
------------

ImWIP heavily relies on CUDA kernels for efficient parallelization. Therfore a CUDA enabeled GPU is required. Furthermore, the following python dependencies are needed, which can be easily installed using conda, and get the package working using the numba/CUDA backend.

- python >= 3.8
- numpy
- scipy
- numba
- pylops
- tqdm

There is also a C++/CUDA backend, which is a bit faster than the numba backend, and it can be accessed from any language that supports a C interface functions. It will automatically compile if the following dependencies are met:

- linux
- cython
- nvcc


Installing ImWIP
----------------

If git is installed, simply run

.. code-block:: console

   $ pip install git+https://github.com/RendersJens/ImWIP.git

Otherwise, download the repository from https://github.com/RendersJens/ImWIP and run pip in the root folder of the project:

.. code-block:: console

    $ pip install .