import numpy as np
import random

class agent:
    def __init__(self, id:int, pw:float, pc:float):
        self.id = id
        self.pw = pw # probability of working
        self.w = 0 # wage
        self.pc = pc # proportion of consumption
        
    def work_decision(self,):
        l = 1 if random.random() < self.pw else 0
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

    def produce(self,):
        pass
    
    def imbalance(self,):
        
    def wage_adjustment(self,):
        pass
    
    def price_adjustment(self,):
        pass
    
class market:
    def __init__(self) -> None:
        self.agents = []
        
    def inflation(self,):
        pass
    
    def unemployment(self,):
        pass
    
    def taxation(self,):
        pass
    
    def 
    

def main():
    N = 100
    
    mar = market()
    F = firm(A=1)
    
    mar.agents = [agent(id=i, pw=np.random.uniform(0.3, 0.7), pc=np.random.uniform(0.2, 0.4)) for i in range(N)]
    
    work_state_history = []
    
    work_state_history.append([a.work_decision() for a in mar.agents]) # t=0 work state
    
    for t in range(1, 100):
    
    
    
    
    