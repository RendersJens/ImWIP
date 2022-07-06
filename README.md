[![DOI](https://zenodo.org/badge/452688446.svg)](https://zenodo.org/badge/latestdoi/452688446)


**ImWIP**: CUDA/C implementations of various warping and adjoint warping and differentiated warping algorithms, with python wrappers.

Features
------------

* Linear and cubic image warping of 2D and 3D images
  * Using a Deformation Vector Field (DVF)
  * Using an affine transformation
* The adjoint action of the above warp functions. Each of these image warps can be seen as a linear operator <img src="https://rawgit.com/RendersJens/ImWIP/main/svgs/73b6888541fd61edff8be10b90799836.svg?invert_in_darkmode" align=middle width=21.46124639999999pt height=22.831056599999986pt/> acting on a vector <img src="https://rawgit.com/RendersJens/ImWIP/main/svgs/be12a978e6d1bafbac7cb59d0d63d3ba.svg?invert_in_darkmode" align=middle width=18.52743584999999pt height=22.831056599999986pt/> that represents the image. The implementation of the adjoint operators <img src="https://rawgit.com/RendersJens/ImWIP/main/svgs/4fbc5a1c55d6970cb45d1978efbb91a0.svg?invert_in_darkmode" align=middle width=31.816839449999986pt height=27.6567522pt/> or <img src="https://rawgit.com/RendersJens/ImWIP/main/svgs/417ad7172885992c7e6d2e8828e45ed5.svg?invert_in_darkmode" align=middle width=29.01833549999999pt height=22.831056599999986pt/> is usefull to solve linear systems involving <img src="https://rawgit.com/RendersJens/ImWIP/main/svgs/73b6888541fd61edff8be10b90799836.svg?invert_in_darkmode" align=middle width=21.46124639999999pt height=22.831056599999986pt/> and to compute analytic derivatives to <img src="https://rawgit.com/RendersJens/ImWIP/main/svgs/be12a978e6d1bafbac7cb59d0d63d3ba.svg?invert_in_darkmode" align=middle width=18.52743584999999pt height=22.831056599999986pt/> of functionals involving <img src="https://rawgit.com/RendersJens/ImWIP/main/svgs/72db7dce8f7422d98674d6fec82b7e9e.svg?invert_in_darkmode" align=middle width=30.85623419999999pt height=22.831056599999986pt/>.
* Analytic derivatives of <img src="https://rawgit.com/RendersJens/ImWIP/main/svgs/a7d50ac9c1d1c38e20129fbd51deb978.svg?invert_in_darkmode" align=middle width=40.18277669999999pt height=24.65753399999998pt/> to <img src="https://rawgit.com/RendersJens/ImWIP/main/svgs/2c81564e7f8fed1d9540fdb33c3889eb.svg?invert_in_darkmode" align=middle width=15.068545799999992pt height=22.831056599999986pt/>, where <img src="https://rawgit.com/RendersJens/ImWIP/main/svgs/a7d50ac9c1d1c38e20129fbd51deb978.svg?invert_in_darkmode" align=middle width=40.18277669999999pt height=24.65753399999998pt/> is a warping operator along rigid or affine motion determined by the vector <img src="https://rawgit.com/RendersJens/ImWIP/main/svgs/2c81564e7f8fed1d9540fdb33c3889eb.svg?invert_in_darkmode" align=middle width=15.068545799999992pt height=22.831056599999986pt/> of rigid or affine motion parameters. This is a basic tool in the development of algorithms that solve for the motion parameters.

As an example, imagine that we want to solve the following system for <img src="https://rawgit.com/RendersJens/ImWIP/main/svgs/2c81564e7f8fed1d9540fdb33c3889eb.svg?invert_in_darkmode" align=middle width=15.068545799999992pt height=22.831056599999986pt/> and <img src="https://rawgit.com/RendersJens/ImWIP/main/svgs/be12a978e6d1bafbac7cb59d0d63d3ba.svg?invert_in_darkmode" align=middle width=18.52743584999999pt height=22.831056599999986pt/>:
```math
BA(t)x = b
```
or similarly, we want to minimize
```math
f(x,t) = \frac{1}{2}\lVert BA(t)x - b \rVert_2^2
```
Here <img src="https://rawgit.com/RendersJens/ImWIP/main/svgs/0c30556449dda1348565ca9dd5bdc7d9.svg?invert_in_darkmode" align=middle width=16.18724414999999pt height=22.831056599999986pt/> can represent data that is the result of moving an unknown image <img src="https://rawgit.com/RendersJens/ImWIP/main/svgs/be12a978e6d1bafbac7cb59d0d63d3ba.svg?invert_in_darkmode" align=middle width=18.52743584999999pt height=22.831056599999986pt/> with unknown affine motion and then applying a known linear transformation <img src="https://rawgit.com/RendersJens/ImWIP/main/svgs/173837d7aca5fffcacc62295b7bf910b.svg?invert_in_darkmode" align=middle width=22.42585124999999pt height=22.831056599999986pt/>. To solve this problem, we need the gradient of <img src="https://rawgit.com/RendersJens/ImWIP/main/svgs/adebe424ff21807c6d2e5cd2f8e0e9b8.svg?invert_in_darkmode" align=middle width=18.94986059999999pt height=22.831056599999986pt/> with respect to <img src="https://rawgit.com/RendersJens/ImWIP/main/svgs/be12a978e6d1bafbac7cb59d0d63d3ba.svg?invert_in_darkmode" align=middle width=18.52743584999999pt height=22.831056599999986pt/> and <img src="https://rawgit.com/RendersJens/ImWIP/main/svgs/2c81564e7f8fed1d9540fdb33c3889eb.svg?invert_in_darkmode" align=middle width=15.068545799999992pt height=22.831056599999986pt/>:
```math
<p align="center"><img src="https://rawgit.com/RendersJens/ImWIP/main/svgs/9c33b3eb3373d2a81c46912d87107f5f.svg?invert_in_darkmode" align=middle width=286.5218862pt height=45.90338775pt/></p>
```
This requires the operators <img src="https://rawgit.com/RendersJens/ImWIP/main/svgs/a7d50ac9c1d1c38e20129fbd51deb978.svg?invert_in_darkmode" align=middle width=40.18277669999999pt height=24.65753399999998pt/>, <img src="https://rawgit.com/RendersJens/ImWIP/main/svgs/aa1fcd88ae076b6ae99e89d7f21ed061.svg?invert_in_darkmode" align=middle width=50.53836974999998pt height=27.6567522pt/> and <img src="https://rawgit.com/RendersJens/ImWIP/main/svgs/ef6f7839fb61f89a0a2213f3bebec133.svg?invert_in_darkmode" align=middle width=44.794652099999986pt height=24.7161288pt/>, which are all provided by this package.

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

`<img src="https://rawgit.com/RendersJens/ImWIP/main/svgs/33050e23b124d58b9c4fea7ed8446448.svg?invert_in_darkmode" align=middle width=562.5129675pt height=78.90410880000002pt/> pip install .`

**Install without pip:**

clone/download the repository and run

`$ python setup.py install`

in the root folder of the project.

Basic usage
-----------
Take a look at `/examples` for basic usage.
