from dsge.functions import derivative


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

    """

    def __init__(self, utility, endowment, time_preference=0.05):
        self.utility = utility
        self.time_preference = time_preference
        self.marginal_utility = derivative(utility)
        self.endowment = endowment

    def get_utility(self, x=None):
        if x is None:
            x = self.endowment
        return self.utility(x)

    def get_marginal_utility(self, x=None):
        if x is None:
            x = self.endowment
        return self.marginal_utility(x)

