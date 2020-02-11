from dsge.agents import Household
from dsge.functions import ces_utility

def household_test():
    print('Household Test:')
    h1 = Household(ces_utility(1), 1)
    return h1.get_marginal_utility()

print(household_test())
