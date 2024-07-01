import numpy as np
import random

class agent:
    def __init__(self, pw:float, pc:float):
        self.id = None
        self.pw = pw # probability of working
        self.w = 0 # wage
        self.pc = pc # proportion of consumption
        
    def work_decision(self,):
        l = 1 if random.random() < self.pw else 0
        return l
    
    def consume(self,):
        pass
    
    def update_wage(self,):

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
    def __init__(self, A):
        self.A = A # universal productivity
        self.G = 0 # quantity of essential goods
        self.P = 0 # price of essential goods

    def produce(self,):
        pass
    
    def wage_adjustment(self,):
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
    
    