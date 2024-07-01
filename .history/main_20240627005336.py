import numpy as np
import random

class agent:
    def __init__(self, id:int, pw:float, pc:float):
        self.id = id
        self.pw = pw # probability of working
        self.w = 0 # wage
        self.pc = pc # proportion of consumption
        self.l = None
        
    def work_decision(self,):
        l = 1 if random.random() < self.pw else 0
        self.l = l
        return l
    
    def consume(self,):
        pass
    
    def update_wage(self,):
        pass
    
class bank:
    def __init__(self, rn:float,):
        self.rn = rn # natural interest rate
        self.deposits = {}
    
    def deposit(self, agent_id, amount):
        if agent_id in self.deposits:
            self.deposits[agent_id] += amount
        else:
            self.deposits[agent_id] = amount
    
    def rate_adjustment(self,):
        pass
    
class firm:
    def __init__(self, A:float, alpha_w:float, alpha_p:float):
        self.A = A # universal productivity
        self.G = 0 # quantity of essential goods
        self.P = 0 # price of essential goods
        self.alpha_w = alpha_w # wage adjustment parameter
        self.alpha_p = alpha_p # price adjustment parameter

    def produce(self, agent_list:list[agent],):
        production = 0
        for a in agent_list:
            if a.l == 1:
                production += 168 * self.A
        self.G += production

        
    def wage_adjustment(self, agent_list:list[agent], imbalance:float):
        '''
        调整工资并不能刺激生产，需要补齐这个逻辑
        '''
        assert len(agent_list) > 0
        sgn = 1 if imbalance > 0 else -1
        
        for a in agent_list:
            a.w *= (1 + sgn * self.alpha_w * abs(imbalance) * np.random.uniform())
    
    def price_adjustment(self, imbalance:float):
        sgn = 1 if imbalance > 0 else -1
        self.P *= (1 + sgn * self.alpha_p * abs(imbalance) * np.random.uniform())
    
class market:
    def __init__(self, ) -> None:
        self.agents:list[agent] = []
        self.bank:bank = None
        
        
    def inflation(self,):
        pass
    
    def unemployment(self,):
        pass
    
    def taxation(self,):
        pass
    
    def total_demand(self, agent_list:list[agent], firm:firm, bank:bank) -> float:
        cnt = 0
        for a in agent_list:
            cnt += (a.pc * bank.deposits[a.id] / firm.P)
        return cnt
    
    def imbalance(self, agent_list:list[agent], firm:firm, bank:bank):
        D = self.total_demand()
        phi_bar = (D - firm.G) / max(D, firm.G)
        return phi_bar

def main():
    N = 100
    T = 12*10 # 12 years
    
    mar = market()
    F = firm(A=1)
    
    mar.agents = [agent(id=i, pw=np.random.uniform(0.3, 0.7), pc=np.random.uniform(0.2, 0.4)) for i in range(N)]
    
    work_state_history = []
    
    work_state_history.append([a.work_decision() for a in mar.agents]) # t=0 work state
    
    for t in range(1, T):
        pass
    
    
    
    