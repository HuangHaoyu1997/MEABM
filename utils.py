import numpy as np


def beta_dist(size=1):
    '''
    truncated beta distribution for generating hourly wage
    '''
    alpha, beta = 1.5, 2
    s = np.random.beta(alpha, beta, size) * 500
    if s <= 500/168:
        return 500/168
    return s[0]

def pay_wage(agent_list:list):
    wages = []
    for a in agent_list:
        if a.l:
            # a.z = a.w * 168   # monthly income
            wages.append(a.w * 168)
        else:
            wages.append(0.)
    return wages