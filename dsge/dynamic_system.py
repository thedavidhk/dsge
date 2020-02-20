#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import interp
from scipy.optimize import fminbound

matplotlib.use('tkagg')


class LinInterp:
    "Provides linear interpolation in one dimension."

    def __init__(self, X, Y):
        """Parameters: X and Y are sequences or arrays
        containing the (x,y) interpolation points.
        """
        self.X, self.Y = X, Y

    def __call__(self, z):
        """Parameters: z is a number, sequence or array.
        This method makes an instance f of LinInterp callable,
        so f(z) returns the interpolation value(s) at z.
        """
        if isinstance(z, int) or isinstance(z, float):
            return interp([z], self.X, self.Y)[0]
        else:
            return interp(z, self.X, self.Y)


def maximize(func, a, b, args=None):
    def g(x):
        return -func(x, *args)

    maximizer = fminbound(g, a, b)
    maximum = func(maximizer, *args)
    return maximum, maximizer


class Dynamic_Optimization(object):
    def __init__(self, objective, s_space, time_discount, transformation):
        self.objective = objective
        self.s_space = s_space
        self.discount = time_discount
        self.transformation = transformation

    # TODO: This will cause an infinite loop. Just keep fore reference
    def value_function(self, action, state, vf_old):
        current_reward = self.objective(action, state)
        future_reward = vf_old(self.transformation(state, action))
        return current_reward + self.discount * future_reward

    # TODO: Generalize action constraint
    def a_space(self, state):
        return np.arange(0, state, 1e-1)

    def bellman(self, vf_old):
        vals = []
        for s in self.s_space:
            # TODO: make this budget constraint more explicit and general
            xmax = s
            vals.append(maximize(self.value_function, 0, xmax, (s, vf_old))[0])
        return LinInterp(self.s_space, vals)

    def value_function_iteration(self, v0=None):
        if v0 is None:

            def v0(x):
                return x

        v = v0
        count = 0
        maxiter = 200
        mse = 1
        tol = 1e-5
        while count < maxiter and mse > tol:
            v_old = v
            v = model.bellman(v)
            mse = sum((v(s_space) - v_old(s_space))**2) / len(s_space)
            if np.mod(count, 10) == 0:
                print("count: %d; mse: %f" % (count, mse))
            count += 1
        print("Convergence after %d iterations" % count)
        return v

    def policy(self, v):
        vals = []
        for s in self.s_space:
            xmax = s
            vals.append(maximize(self.value_function, 0, xmax, (s, v))[1])
        return LinInterp(self.s_space, vals)


if __name__ == "__main__":

    delta = 0.1
    alpha = 0.8
    beta = 0.95
    s_space = np.linspace(1e-1, 20, 100)

    def u(x, k):
        return np.log(x)

    def s(x, k_old):
        return k_old * (1 - delta) + k_old**alpha - x

    model = Dynamic_Optimization(u, s_space, beta, s)
    v_star = model.value_function_iteration()

    # Optimal policy:
    pi_star = model.policy(v_star)

    # Show in plot:
    fig, ax = plt.subplots()
    ax.set_xlim(s_space.min(), s_space.max())
    ax.plot(s_space, pi_star(s_space))
    plt.show()

    # plt.show()
