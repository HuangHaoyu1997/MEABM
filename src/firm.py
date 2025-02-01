import random
import numpy as np
from src.agent import agent

class firm:
    def __init__(self, 
                 A:float, 
                 alpha_w:float, 
                 alpha_p:float,
                 alpha_c:float, 
                 init_good:float,
                 init_cap:float,
                 k_labor:float,
                 k_capital:float,
                 ):
        self.A = A               # universal productivity
        self.G = init_good       # quantity of essential goods
        self.P = 0               # price of essential goods
        self.alpha_w = alpha_w   # wage adjustment parameter
        self.alpha_p = alpha_p   # price adjustment parameter
        self.alpha_c = alpha_c   # capital adjustment parameter
        self.capital = init_cap  # initial capital
        
        self.k_labor = k_labor   
        self.k_capital = k_capital
        self.cap4product = 5e-5 * init_cap  # init capital for production

    def produce(self, agent_list:list[agent], heterogeneity:bool=False):
        '''
        production of essential goods
        
        '''
        # random production
        workers = sum([a.l for a in agent_list])
        if not heterogeneity:
            production = 168 * self.A * (1 + random.random() * 0.05) * \
                                    (workers**self.k_labor) * \
                                    (self.cap4product ** self.k_capital)
        else:
            production = sum([a.l * a.A**self.k_labor * (1 + random.random() * 0.05) for a in agent_list]) * \
                                    (self.cap4product ** self.k_capital)
        
        self.G += production
        self.capital -= self.cap4product
        return production
    
    def cap_adjustment(self, imbalance:float):
        '''
        imbalance>0, 需求 > 产量, 增加资本
        imbalance<0, 需求 < 产量, 减少资本
        '''
        sgn = 1 if imbalance > 0 else -1
        self.cap4product *= (1 + sgn * self.alpha_c * abs(imbalance) * np.random.uniform())
        
    def wage_adjustment(self, agent_list:list[agent], imbalance:float):
        '''
        imbalance>0, 需求 > 产量, 提高工资
        imbalance<0, 需求 < 产量, 降低工资
        
        只调整当期就业工人的工资
        '''
        assert len(agent_list) > 0
        sgn = 1 if imbalance > 0 else -1
        for a in agent_list:
            a.w *= (1 + sgn * self.alpha_w * abs(imbalance) * np.random.uniform())
    
    def price_adjustment(self, imbalance:float):
        '''
        imbalance>0, 需求 > 产量, 提高价格
        imbalance<0, 需求 < 产量, 降低价格
        '''
        sgn = 1 if imbalance > 0 else -1
        self.P *= (1 + sgn * self.alpha_p * abs(imbalance) * np.random.uniform())
        # return self.P
    
    def pay_wage(self, agent_list:list):
        wages = []
        for a in agent_list:
            if a.l:
                # a.z = a.w * 168   # monthly income
                wages.append(a.w * 168)
            else:
                wages.append(0.)
        self.capital -= sum(wages)
        return wages