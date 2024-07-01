import numpy as np


def beta_dist(size=1):
    '''
    truncated beta distribution for generating hourly wage
    '''
    alpha, beta = 1.5, 2
    s = np.random.beta(alpha, beta, size) * 500
    if s <= 500:
        s = [500]
    return s