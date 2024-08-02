import random
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

class agent:
    '''
    homogeneous worker/consumer agent.
    working skill is same (unit one) for all agents.
    '''
    def __init__(self, id:int, pw:float, pc:float, gamma:float, beta:float):
        self.id = id
        self.pw = pw         # probability of working
        self.w = beta_dist() # hourly wage
        self.z = 0           # monthly income
        self.pc = pc         # proportion of consumption
        self.l = 0           # work state (0: unemployed, 1: employed)
        self.gamma = gamma
        self.beta = beta
        
        
    def work_decision(self,):
        self.l = 1 if random.random() < self.pw else 0
        return self.l
    
    def work_adjustment(self, deposit:float, rate:float):
        '''
        adjust the probability of working based on deposit and interest rate
        '''
        self.pw = pow((self.z + 1e-5) / (deposit + deposit*rate + 1e-5), self.gamma)
        # print(self.id, self.z, deposit, "pw:", self.pw)
        return self.pw
    
    def consume_adjustment(self, P:float, deposit:float):
        '''
        adjust the proportion of consumption based on the price and deposit
        '''
        self.pc = pow(P / (deposit + self.z + 1e-5), self.beta)
        return self.pc