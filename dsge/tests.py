from dsge.agents import Household
from dsge.functions import ces_utility
import numpy as np


def household_test():
    print('Household Test:')
    h1 = Household(ces_utility(1), 1)
    x_stream1 = [1, 2, 3, 4, 5]
    x_stream2 = [3, 3, 3, 3, 3]
    x_stream3 = [5, 4, 3, 2, 1]
    x_stream_list = [x_stream1, x_stream2, x_stream3]
    print(h1.utility_pv(x_stream1))
    print(h1.utility_pv(x_stream2))
    print(h1.utility_pv(x_stream3))
    print("This stream maximizes the present lifetime utility:")
    print(h1.max_util(x_stream_list))


def bellman_test():

    def hh_budget_constraint(y, c):
        return (y-c)**0.4

    h1 = Household(ces_utility(1), 1, hh_budget_constraint)
    v_0 = 2*np.log(h1._opt.grid)+4
    h1._opt.vfi(v_0, n=4)


# household_test()
# import pdb; pdb.set_trace()
bellman_test()
