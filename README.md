[![DOI](https://zenodo.org/badge/452688446.svg)](https://zenodo.org/badge/latestdoi/452688446)


**ImWIP**: CUDA/C implementations of various warping and adjoint warping and differentiated warping algorithms, with python wrappers.

Features
------------

* Linear and cubic image warping of 2D and 3D images
  * Using a Deformation Vector Field (DVF)
  * Using an affine transformation
* The adjoint action of the above warp functions. Each of these image warps can be seen as a linear operator <img src="https://rawgit.com/RendersJens/ImWIP/None/svgs/53d147e7f3fe6e47ee05b88b166bd3f6.svg?invert_in_darkmode" align=middle width=12.32879834999999pt height=22.465723500000017pt/> acting on a vector <img src="https://rawgit.com/RendersJens/ImWIP/None/svgs/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode" align=middle width=9.39498779999999pt height=14.15524440000002pt/> that represents the image. The implementation of the adjoint operators <img src="https://rawgit.com/RendersJens/ImWIP/None/svgs/99f7812af37ee7004df7177a1e821ec5.svg?invert_in_darkmode" align=middle width=21.86251649999999pt height=27.6567522pt/> or <img src="https://rawgit.com/RendersJens/ImWIP/None/svgs/6b76fc0b9cd7cb371b27ad5803620550.svg?invert_in_darkmode" align=middle width=19.063992749999993pt height=22.63846199999998pt/> is usefull to solve linear systems involving <img src="https://rawgit.com/RendersJens/ImWIP/None/svgs/53d147e7f3fe6e47ee05b88b166bd3f6.svg?invert_in_darkmode" align=middle width=12.32879834999999pt height=22.465723500000017pt/> and to compute analytic derivatives to <img src="https://rawgit.com/RendersJens/ImWIP/None/svgs/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode" align=middle width=9.39498779999999pt height=14.15524440000002pt/> of functionals involving <img src="https://rawgit.com/RendersJens/ImWIP/None/svgs/bbb2565155df2f2e483c15107e8505b1.svg?invert_in_darkmode" align=middle width=21.723786149999988pt height=22.465723500000017pt/>.
* Analytic derivatives of <img src="https://rawgit.com/RendersJens/ImWIP/None/svgs/5f1ace7f43d147d16685246df2a801c6.svg?invert_in_darkmode" align=middle width=31.05032864999999pt height=24.65753399999998pt/> to <img src="https://rawgit.com/RendersJens/ImWIP/None/svgs/4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode" align=middle width=5.936097749999991pt height=20.221802699999984pt/>, where <img src="https://rawgit.com/RendersJens/ImWIP/None/svgs/5f1ace7f43d147d16685246df2a801c6.svg?invert_in_darkmode" align=middle width=31.05032864999999pt height=24.65753399999998pt/> is a warping operator along rigid or affine motion determined by the vector <img src="https://rawgit.com/RendersJens/ImWIP/None/svgs/4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode" align=middle width=5.936097749999991pt height=20.221802699999984pt/> of rigid or affine motion parameters. This is a basic tool in the development of algorithms that solve for the motion parameters.

As an example, imagine that we want to solve the following system for <img src="https://rawgit.com/RendersJens/ImWIP/None/svgs/4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode" align=middle width=5.936097749999991pt height=20.221802699999984pt/> and <img src="https://rawgit.com/RendersJens/ImWIP/None/svgs/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode" align=middle width=9.39498779999999pt height=14.15524440000002pt/>:
<p align="center"><img src="https://rawgit.com/RendersJens/ImWIP/None/svgs/87a280c0a9ff0b90c7f09a08993028e8.svg?invert_in_darkmode" align=middle width=82.71114885pt height=16.438356pt/></p>

or similarly, we want to minimize
<p align="center"><img src="https://rawgit.com/RendersJens/ImWIP/None/svgs/9116c69b4bf34c5010af96ea5559072e.svg?invert_in_darkmode" align=middle width=183.19751835pt height=32.990165999999995pt/></p>

Here <img src="https://rawgit.com/RendersJens/ImWIP/None/svgs/4bdc8d9bcfb35e1c9bfb51fc69687dfc.svg?invert_in_darkmode" align=middle width=7.054796099999991pt height=22.831056599999986pt/> can represent data that is the result of moving an unknown image <img src="https://rawgit.com/RendersJens/ImWIP/None/svgs/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode" align=middle width=9.39498779999999pt height=14.15524440000002pt/> with unknown affine motion and then applying a known linear transformation <img src="https://rawgit.com/RendersJens/ImWIP/None/svgs/61e84f854bc6258d4108d08d4c4a0852.svg?invert_in_darkmode" align=middle width=13.29340979999999pt height=22.465723500000017pt/>. To solve this problem, we need the gradient of <img src="https://rawgit.com/RendersJens/ImWIP/None/svgs/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode" align=middle width=9.81741584999999pt height=22.831056599999986pt/> with respect to <img src="https://rawgit.com/RendersJens/ImWIP/None/svgs/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode" align=middle width=9.39498779999999pt height=14.15524440000002pt/> and <img src="https://rawgit.com/RendersJens/ImWIP/None/svgs/4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode" align=middle width=5.936097749999991pt height=20.221802699999984pt/>:
<p align="center"><img src="https://rawgit.com/RendersJens/ImWIP/None/svgs/1f80b08b5cd75529f481e70861365fbb.svg?invert_in_darkmode" align=middle width=286.5218862pt height=45.90338775pt/></p>
This requires the operators <img src="https://rawgit.com/RendersJens/ImWIP/None/svgs/5f1ace7f43d147d16685246df2a801c6.svg?invert_in_darkmode" align=middle width=31.05032864999999pt height=24.65753399999998pt/>, <img src="https://rawgit.com/RendersJens/ImWIP/None/svgs/31eb1f570ea7331e2666697ae4cae1d5.svg?invert_in_darkmode" align=middle width=41.405921699999986pt height=27.6567522pt/> and <img src="https://rawgit.com/RendersJens/ImWIP/None/svgs/51f1bcfdd671038e2035d5c0bf517dbc.svg?invert_in_darkmode" align=middle width=35.662204049999986pt height=24.7161288pt/>, which are all provided by this package.

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

`<img src="https://rawgit.com/RendersJens/ImWIP/None/svgs/33050e23b124d58b9c4fea7ed8446448.svg?invert_in_darkmode" align=middle width=562.5129675pt height=78.90410880000002pt/> pip install .`

**Install without pip:**

clone/download the repository and run

`$ python setup.py install`

in the root folder of the project.

Basic usage
-----------
Take a look at `/examples` for basic usage.
