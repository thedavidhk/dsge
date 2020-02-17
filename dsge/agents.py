from dsge.functions import derivative
from dsge.optimization import Opt_Problem
import math


class Household:
    """Representative household in the economy

    Attributes
    ----------
    utility : function
        Utility function (e.g. dsge.functions.ces_utility).
    marginal_utility : function
        First derivative of the utility function.
    endowment : int
        Initial wealth of the individual household.
    budget : function
        Intertemporal budget constraint of the household

    """

    def __init__(self, utility, endowment, budget, time_preference=0.05):
        self._utility = utility
        self._time_preference = time_preference
        self._marginal_utility = derivative(utility)
        self._endowment = endowment
        self._budget = budget
        self._opt = Opt_Problem(utility, budget, time_preference)

    def utility(self, x=None):
        if x is None:
            x = self._endowment
        return self._utility(x)

    def marginal_utility(self, x=None):
        if x is None:
            x = self._endowment
        return self._marginal_utility(x)

    def utility_pv(self, x_stream):
        rho = self._time_preference
        e = math.e
        u = sum([self.utility(x) / (1 + rho)**t
                 for t, x in enumerate(x_stream)])
        return u

    def max_util(self, x_stream_list):
        return max(x_stream_list, key=self.utility_pv)

