#!/usr/bin/env python
# import matplotlib
from functools import lru_cache
import matplotlib.pyplot as plt
import numpy as np
from numpy import interp
import pickle
from scipy.optimize import fminbound

# matplotlib.use('tkagg')


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
        future_reward = vf_old(self.transformation(action, state))
        return current_reward + self.discount * future_reward

    def policy(self, v):
        """Determine the optimal policy given a value function

        Args:
            v (callable): Value function.

        Returns:
            callable: Optimal policy function.
        """
        vals = []
        for s in self.s_space:
            xmax = s**0.5
            vals.append(maximize(self.value, 1e-2, xmax, (s, v))[1])
        return LinInterp(self.s_space, vals)

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
            xmax = s**0.5
            vals.append(maximize(self.value, 1e-2, xmax, (s, vf_old))[0])
        return LinInterp(self.s_space, vals)

    @lru_cache(maxsize=8)
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
                return 0

        v = v0
        msd = tol + 1  # Mean squared difference. Initialized > tol.
        for i in range(maxiter):
            v_old = v
            v = self.bellman(v)
            msd = sum(
                (v(self.s_space) - v_old(self.s_space))**2) / len(self.s_space)
            if (i + 1) % 10 == 0:
                print("count: %d; mse: %f" % (i + 1, msd))
            if msd < tol:
                print("Convergence after %d iterations." % (i + 1))
                break
            if i == maxiter - 1:
                print("Convergence not reached after %d iterations." % maxiter)
        return v

    def policy_value(self, pi, tol=1e-5):
        def v(x):
            return 0

        maxiter = 100
        for i in range(maxiter):
            v_old = v
            vals = []
            for s in self.s_space:
                action = pi(s)
                vals.append(self.value(action, s, v_old))
            v = LinInterp(self.s_space, vals)
            msd = sum(
                (v(self.s_space) - v_old(self.s_space))**2) / len(self.s_space)
            if msd < tol:
                # value converged
                break
        return v

    def policy_iteration(self, pi0=None, tol=1e-5, maxiter=200):
        """Policy iteration.

        Args:
            p0 (callable, optional): Inital guess for the policy function.
            tol (float): Tolerance for convergence criterion. If the mean
                squared difference between two iterations is less than tol,
                break the loop and consider the value function converged.

        Returns:
            callable: Converged policy function.
        """
        if pi0 is None:

            def pi0(x):
                return 0

        pi = pi0
        msd = tol + 1  # Mean squared difference. Initialized > tol.
        for i in range(maxiter):
            v_pi_old = self.policy_value(pi)
            pi_new = self.policy(v_pi_old)
            msd = sum((pi_new(self.s_space) - pi(self.s_space))**2) / len(
                self.s_space)
            if (i + 1) % 10 == 0:
                print("count: %d; mse: %f" % (i + 1, msd))
            if msd < tol:
                print("Convergence after %d iterations." % (i + 1))
                break
            if i == maxiter - 1:
                print("Convergence not reached after %d iterations." % maxiter)
            pi = pi_new
        return pi


if __name__ == "__main__":

    delta = 0.1
    alpha = 0.5
    beta = 0.95
    n = 0.05
    kap = np.linspace(1e-1, 20, 200)
    s_space = kap

    def f(kap, lab, alpha=0.5):
        return kap**alpha * lab**(1 - alpha)

    def u(x, s):
        return np.log(x)

    def k_new(c, k):
        return k * (1 - delta) + f(k, 1) - c

    model = Dynamic_Optimization(u, s_space, beta, k_new)

    try:
        with open("piter.pyobj", "rb") as piter:
            c_star = pickle.load(piter)
        print("File loaded")
    except FileNotFoundError:
        c_star = model.policy_iteration(tol=1e-6)
        with open("piter.pyobj", "wb") as piter:
            pickle.dump(c_star, piter)
        print("File not found. Created new cache file piter.pyobj.")

    # Optimal policy:
    # c_star = model.policy(v_star)

    # Simulate time series
    T = 100
    k = []
    y = []
    c = []
    k.append(1)

    for t in range(T):
        y.append(f(k[t], 1))
        c.append(c_star(k[t]))
        if t < T - 1:
            k.append(k_new(c[t], k[t]))

    fig, ax = plt.subplots()
    ax.plot(range(T), y, label="y")
    ax.plot(range(T), c, label="c")
    ax.plot(range(T), k, label="k")
    ax.legend(loc='lower right')
    plt.show()
