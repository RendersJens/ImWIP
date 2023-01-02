"""
:file:      solvers.py
:brief:     a small collection of solvers, which are suitable for
            most inverse problems involving image warps.
:author:    Jens Renders
"""


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
    """ projected gradient descent with BB step size and bounds
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
            callback(x)

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
    """ A split version of projected gradient descent with BB step size and bounds
        we use different stepsizes for different types of variables
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
            callback(*x)

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