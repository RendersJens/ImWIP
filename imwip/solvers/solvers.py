"""
:file:      solvers.py
:brief:     a small collection of solvers, which are suitable for
            most inverse problems involving image warps.
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
from tqdm import tqdm


def _line_search(f, fx, grad, x, d, t=0.5, c=1e-4, a=1.0, max_iter=20, verbose=False):
    m = np.dot(grad, d/np.linalg.norm(d))
    iterations = 0
    new_f = f(x + a*d)
    while iterations < max_iter and new_f > fx + a*c*m:
        if verbose:
            print(iterations, new_f, fx + a*c*m)
        a *= t
        new_f = f(x + a*d)
        iterations += 1
    return a


def _get_split_stepsize(x, grad, f=None, init_step=None, line_search_iter=None, verbose=False):
    if init_step is not None:
        a = [0]*len(x)
        for i in range(len(x)):
            n_grad = np.linalg.norm(grad[i])
            if n_grad > 0:
                a[i] = init_step[i]/np.linalg.norm(grad[i])
    else:
        if f is None:
            if line_search_iter is not None:
                raise ValueError("f should be given for line_search")
            a = [1/np.linalg.norm(np.concatenate(grad))]*len(x)
        else:
            if line_search_iter is None:
                line_search_iter = 20
            a = [0]*len(x)
            for i in range(len(x)):
                n_grad = np.linalg.norm(grad[i])
                if n_grad > 0:
                    def f_partial(xi, i):
                        x_edit = x[:]
                        x_edit[i] = xi
                        return f(*x_edit)
                    a[i] = _line_search(
                        lambda xi: f_partial(xi, i),
                        f_partial(x[i], i),
                        grad[i],
                        x[i],
                        grad[i],
                        a=1/n_grad,
                        max_iter=line_search_iter,
                        verbose=verbose
                        )
    return a



def barzilai_borwein(grad_f,
                     x0,
                     f=None,
                     line_search_iter=None,
                     init_step=None,
                     bounds=None,
                     max_iter=100,
                     verbose=True,
                     callback=None):
    """
    Minimizes a function f using projected gradient descent with the step size of
    :cite:t:`barzilai1988two` and bounds. The BB stepsize is only defined from the second
    iteration on. Therefore, the initial step size has to be computed by some other method.
    By default it will be 1, but it can be specified or it can be searched with a line
    search.

    .. note::
        the function f itself is not a required arguement for this optimizer. It is
        only needed if you want to use a line search for the intial step size.

    :param grad_f: gradient of the function to be minimized. It should return a 1D
        :class:`numpy.ndarray` of the same size as its input (which is the same size as x0).
    :param x0: initial guess for the minimum
    :param f: function to be minimized, used for the line search for the initial step size.
        Not used if line_search_iter is None. It should return a float
    :param line_search_iter: How many iterations to perform in the line search for the
        initial stepsize. Leave empty for no line search.
    :param init_step: user specified initial stepsize. Leave empty to use line search.
    :param bounds: minimum and maximum constraints on the variables. Leave empty for no
        bounds, and use ``numpy.inf`` or ``-numpy.inf`` to bound only from one side.
    :param max_iter: maximum number of iterations to perform
    :param verbose: whether to show a progress bar, defaults to False.
    :param callback: If given, this function will be called each iteration. The current estimate of
        the minimum will be passed as arguement.

    :type grad_f: callable
    :type x0: :class:`numpy.ndarray`
    :type f: callable, optional
    :type line_search_iter: int, optional
    :type init_step: float, optional
    :type bounds: tuple of floats or tuple of sequences of floats, optional
    :type max_iter: int, optional
    :type verbose: bool, optional
    :type callback: callable, optional

    :return: the minimum x, same size as x0
    :rtype: :class:`numpy.ndarray`
    """
    x = x0

    # gradient at begin position
    grad = grad_f(x)

    # initial stepsize
    if init_step is not None:
        a = init_step/np.linalg.norm(grad)
    else:
        if f is None:
            if line_search_iter is not None:
                raise ValueError("f should be given for line_search")
            a = 1/np.linalg.norm(grad)
        else:
            if line_search_iter is None:
                line_search_iter = 20
            a = _line_search(f, f(x), grad, x, grad,
                a=1/np.linalg.norm(grad),
                max_iter=line_search_iter,
                verbose=verbose)

    # gradient descent step
    xp = x
    x = x - a*grad

    gradp = grad
    grad = grad_f(x)
    xp = x
    x = x - a*grad

    # now that we have our previous value xp
    # we can start iterating
    if verbose:
        loop = tqdm(range(max_iter))
    else:
        loop = range(max_iter)
    for i in loop:

        # gradient at previous solution
        gradp = grad
        grad = grad_f(x)

        if callback is not None:
            callback(x,i)

        # BB step size
        grad_diff = grad - gradp
        a = abs(np.dot(x-xp, grad_diff))/np.dot(grad_diff, grad_diff)

        # new solution: gradient descent step
        xp = x
        x = x - a*grad

        # apply the bounds
        if bounds is not None:
            x = np.clip(x, *bounds)

    return x


def split_barzilai_borwein(
        grad_f,
        x0,
        f=None,
        line_search_iter=None,
        init_step=None,
        bounds=None,
        max_iter=100,
        verbose=True,
        callback=None
    ):
    """
    A split version of :py:func:`barzilai_borwein`. Instead of using a single stepsize
    for all variables, this function allows to group the variables and use a separate
    stepsize for each group.

    :param grad_f: gradient of the function to be minimized. It should return a list of 1D
        :class:`numpy.ndarray` (one for each group of variables) of the same size as its
        input (which is the same size as x0).
    :param x0: initial guess for the minimum, one array per group of variables.
    :param f: function to be minimized, used for the line search for the initial step size.
        Not used if line_search_iter is None. It should return a float
    :param line_search_iter: How many iterations to perform in the line search for the
        initial stepsize. Leave empty for no line search.
    :param init_step: user specified initial stepsize. Leave empty to use line search.
    :param bounds: minimum and maximum constraints on the variables. Leave empty for no
        bounds, and use ``numpy.inf`` or ``-numpy.inf`` to bound only from one side.
    :param max_iter: maximum number of iterations to perform
    :param verbose: whether to show a progress bar, defaults to False.
    :param callback: If given, this function will be called each iteration. The current estimate of
        the minimum will be passed as arguement.

    :type grad_f: callable
    :type x0: list of :class:`numpy.ndarray`
    :type f: callable, optional
    :type line_search_iter: int, optional
    :type init_step: float, optional
    :type bounds: list of bounds as specified in :py:func:`barzilai_borwein`
    :type max_iter: int, optional
    :type verbose: bool, optional
    :type callback: callable, optional

    :return: the minimum x
    :rtype: list of :class:`numpy.ndarray`
    """
    x = [np.array(var) for var in x0]

    # gradient at begin position
    grad = grad_f(*x)

    # initial stepsize
    a = _get_split_stepsize(
        x,
        grad,
        f=f,
        init_step=init_step,
        line_search_iter=line_search_iter,
        verbose=verbose
    )

    # gradient descent step
    xp = x[:]
    for i in range(len(x)):
        x[i] = x[i] - a[i]*grad[i]

    # another gradient descent step
    gradp = grad
    grad = grad_f(*x)
    a = _get_split_stepsize(
        x,
        grad,
        f=f,
        init_step=init_step,
        line_search_iter=line_search_iter,
        verbose=verbose
    )
    xp = x[:]
    for i in range(len(x)):
        x[i] = x[i] - a[i]*grad[i]

    # now that we have our previous value xp
    # we can start iterating
    if verbose:
        loop = tqdm(range(max_iter))
    else:
        loop = range(max_iter)
    for i in loop:

        # gradient at previous solution
        gradp = grad
        grad = grad_f(*x)

        if callback is not None:
            callback(*x,i)

        # independent BB step sizes
        for i in range(len(x)):
            grad_diff = grad[i] - gradp[i]
            a[i] = abs(np.dot(x[i]-xp[i], grad_diff))/np.dot(grad_diff, grad_diff)

        # new solution: gradient descent step
        xp = x[:]
        for i in range(len(x)):
            x[i] = x[i] - a[i]*grad[i]

        # apply the bounds
        if bounds is not None:
            for i in range(len(x)):
                if bounds[i] is not None:
                    x[i] = np.clip(x[i], *bounds[i])

    return x