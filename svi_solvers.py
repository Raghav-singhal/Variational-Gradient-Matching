"""Implements O.D.E solvers"""
import numpy as np



def euler(x, f, eps, params=None):
    if params is None:
        return x + eps * f(x)
    return x + eps * f(x, params)


def rk4(x, f, eps, params=None):
    if params is None:
        k1 = f(x)
        k2 = f(x + 0.5 * eps * k1)
        k3 = f(x + 0.5 * eps * k2)
        k4 = f(x + eps * k3)
    else:
        k1 = f(x, params)
        k2 = f(x + 0.5 * eps * k1, params)
        k3 = f(x + 0.5 * eps * k2, params)
        k4 = f(x + eps * k3, params)
    return x + (eps / 6) * (k1 + 2*k2 + 2*k3 + k4)


def solve(x0, F, T, eps=1e-3, method_name='rk4', params=None):
    if method_name == 'rk4':
        method = rk4
    elif method_name == 'euler':
        method = euler

    x0 = np.asarray(x0)
    path = [x0]

    N = int(T / eps)            # time steps

    for i in range(N - 1):
        if params is None:
            x0 = method(x0, F, eps)
        else:
            x0 = method(x0, F, eps, params)

        path.append(x0)

    return np.array(path)
