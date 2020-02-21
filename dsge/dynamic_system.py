#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import interp
from scipy.optimize import fminbound

matplotlib.use('tkagg')


class LinInterp:
    """Provides linear interpolation in one dimension.

    Attributes:
        X (iterable): Points in first dimension.
        Y (iterable): Points in second dimension.
    """

    def __init__(self, X, Y):
        """__init__

        Args:
            X (iterable): Points in first dimension.
            Y (iterable): Points in second dimension.
        """
        self.X, self.Y = X, Y

    def __call__(self, z):
        """__call__

        This method makes an instance f of LinInterp callable,
        so f(z) returns the interpolation value(s) at z.

        Args:
            z (float or Iterable): Point where to interpolate.
        """
        if isinstance(z, int) or isinstance(z, float):
            return interp([z], self.X, self.Y)[0]
        else:
            return interp(z, self.X, self.Y)


def maximize(func, a, b, args=None):
    """Maximize function value between a and b

    Args:
        func (callable): Function to maximize (must return a scalar).
            The first argument of func will be used to maximize.
        a (float): Lower bound.
        b (float): Upper bound.
        args (tuple): Further arguments to be passed to func.

    Returns:
        tuple of float: Maximum, maximizer
    """
    def g(x, *args):
        return -func(x, *args)

    maximizer = fminbound(g, a, b, args)
    maximum = func(maximizer, *args)
    return maximum, maximizer


class Dynamic_Optimization(object):
    """Dynamic_Optimization

    Attributes:
        objective (callable): Objective function.
        s_space (np.meshgrid): State space.
        discount (float): Discount factor (0 < time_discount < 1).
        transformation (callable): Transformation to next state given action.
        """
    def __init__(self, objective, s_space, time_discount, transformation):
        """__init__

        Args:
            objective (callable): Objective function.
            s_space (np.meshgrid): State space.
            time_discount (float): Discount factor (0 < time_discount < 1).
            transformation (callable): Transformation to next state given
                action.
        """
        self.objective = objective
        self.s_space = s_space
        self.discount = time_discount
        self.transformation = transformation

    def value(self, action, state, vf_old):
        """Value function

        The value function assigns a value to a projected state given an action
        based on a value function vf_old.

        Args:
            action (float or :obj:`list` of float): Action.
            state (float or :obj:`list` of float): Current state.
            vf_old (callable): Value function of previous iteration.

        Returns:
            float: Value of the current state and action based on vf_old.
        """
        current_reward = self.objective(action, state)
        future_reward = vf_old(self.transformation(state, action))
        return current_reward + self.discount * future_reward

    def bellman(self, vf_old):
        """Bellman operator

        The Bellman operator takes a value function as an input and calculates
        a new value function by maximizing the value of the projected state
        given over the action space for all states in the state space.

        Args:
            vf_old (callable): Value function of previous iteration.

        Returns:
            callable: New value function
        """
        vals = []
        for s in self.s_space:
            # TODO: make this budget constraint more explicit and general
            xmax = s[0]
            vals.append(maximize(self.value, 0, xmax, (s, vf_old))[0])
        return LinInterp(self.s_space, vals)

    def value_function_iteration(self, v0=None, tol=1e-5, maxiter=200):
        """Value function iteration

        Value function iteration

        Args:
            v0 (callable, optional): Inital guess for the value function.
            tol (float): Tolerance for convergence criterion. If the mean
                squared difference between two iterations is less than tol,
                break the loop and consider the value function converged.

        Returns:
            callable: Converged value function.
        """
        if v0 is None:

            def v0(x):
                return x

        v = v0
        count = 0
        msd = tol + 1  # Mean squared difference. Initialized > tol.
        while count < maxiter and msd > tol:
            v_old = v
            v = model.bellman(v)
            msd = sum((v(s_space) - v_old(s_space))**2) / len(s_space)
            if count % 10 == 0:
                print("count: %d; mse: %f" % (count, msd))
            count += 1
        if count <= 200:
            print("Convergence after %d iterations." % count)
        else:
            print("Convergence not reached after %d iterations." % maxiter)
        return v

    def policy(self, v):
        """Determine the optimal policy given a value function

        Args:
            v (callable): Value function.

        Returns:
            callable: Optimal policy function.
        """
        vals = []
        for s in self.s_space:
            xmax = s
            vals.append(maximize(self.value_function, 0, xmax, (s, v))[1])
        return LinInterp(self.s_space, vals)


if __name__ == "__main__":

    delta = 0.1
    alpha = 0.4
    beta = 0.90
    n = 0.05
    kap = np.linspace(1e-1, 20, 100)
    lab = np.linspace(1e-1, 20, 100)
    s_space = [kap, lab]

    def f(kap, lab, alpha=0.5):
        return kap**alpha * lab**(1 - alpha)

    def u(x, s):
        return np.log(x)

    def s(x, s_old):
        k_old = s_old[0]
        l_old = s_old[1]
        kap = k_old * (1 - delta) + f(k_old, l_old) - x
        lab = l_old * (1 + n)
        return [kap, lab]

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
