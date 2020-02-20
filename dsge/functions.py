import math
from scipy.optimize import minimize_scalar


def ces_utility(sigma):
    """Constant elasticity of (intertemporal) substitution utility function

    Paramters
    ---------
    sigma : float
        Elasticity of intertemporal substitution

    Return
    ------
    util : function
        Utility function

    """
    def util(consumption):
        c = consumption
        if sigma == 1:
            return math.log(c)
        return (c**(1 - sigma) - 1) / (1 - sigma)
    return util


def cobb_douglas1(alpha):
    def f(s, c):
        return s**alpha - c
    return f


def derivative(function, location=None, argument=0, precision=0.000001):
    d_x = precision
    if location is None:
        def deriv(x):
            d_y = function(x + d_x) - function(x)
            return d_y / d_x
        return deriv
    d_y = function(location + d_x) - function(location)
    return d_y / d_x


def maximize(g, a, b, args):
    def objective(x):
        return -g(x, *args)

    result = minimize_scalar(objective, bounds=(a, b), method='bounded')
    maximizer, maximum = result.x, -result.fun
    return maximizer, maximum
