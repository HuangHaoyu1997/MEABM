import random
import numpy as np
from src.agent import agent

class firm:
    def __init__(self, A:float, alpha_w:float, alpha_p:float):
        self.A = A # universal productivity
        self.G = 0 # quantity of essential goods
        self.P = 0 # price of essential goods
        self.alpha_w = alpha_w # wage adjustment parameter
        self.alpha_p = alpha_p # price adjustment parameter

    def produce(self, agent_list:list[agent],):
        '''
        production of essential goods
        '''
        # random production
        workers = sum([a.l for a in agent_list])
        production = 168 * self.A * (1 + random.random() * 0.05) * (workers**0.9)
        # for a in agent_list:
        #     if a.l == 1: production += 168 * self.A
        self.G += production
        return production
    
    def wage_adjustment(self, agent_list:list[agent], imbalance:float):
        assert len(agent_list) > 0
        sgn = 1 if imbalance > 0 else -1
        for a in agent_list:
            a.w *= (1 + sgn * self.alpha_w * abs(imbalance) * np.random.uniform())
    
    def price_adjustment(self, imbalance:float):
        sgn = 1 if imbalance > 0 else -1
        self.P *= (1 + sgn * self.alpha_p * abs(imbalance) * np.random.uniform())
        return self.P
    
    def pay_wage(self, agent_list:list):
        wages = []
        for a in agent_list:
            if a.l:
                # a.z = a.w * 168   # monthly income
                wages.append(a.w * 168)
            else:
                wages.append(0.)
        return wages