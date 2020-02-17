import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from interpolation import interp

from dsge.functions import maximize


class Opt_Problem:

    def __init__(self, u, f, β=0.95, grid_max=4, grid_size=120):

        self.u, self.f, self.β = u, f, β

        self.grid = np.linspace(1e-4, grid_max, grid_size)

    def objective(self, c, s, v_array):

        u, f, β = self.u, self.f, self.β

        v = lambda x: interp(self.grid, v_array, x)

        return u(c) + β * v(f(s, c))

    def T(self, v):

        v_new = np.empty_like(v)
        v_greedy = np.empty_like(v)

        for i in range(len(self.grid)):
            s = self.grid[i]

            c_star, v_max = maximize(self.objective, 1e-10, s, (s, v))
            v_new[i] = v_max
            v_greedy[i] = c_star

        return v_greedy, v_new

    def vfi(self, v_0, n=20):

        v = v_0
        fig, ax = plt.subplots()

        ax.plot(self.grid, v, color=plt.cm.jet(0),
                lw=2, alpha=0.6, label='Initial condition')

        for i in range(n):
            v_greedy, v = self.T(v)
            ax.plot(self.grid, v, color=plt.cm.jet(i / n), lw=2, alpha=0.4)

        ax.legend()
        ax.set(ylim=(0, 1.5), xlim=(np.min(self.grid), np.max(self.grid)))
        plt.show()

