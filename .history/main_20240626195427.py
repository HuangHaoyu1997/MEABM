import numpy as np
import random

class agent:
    def __init__(self, pw:float, pc:float):
        self.pw = pw # probability of working
        self.w = 0 # wage
        self.pc = pc # proportion of consumption
        
    def work_decision(self,):
        l = 1 if random.random() < self.pw else 0
        return l
    
    def consume(self,
class bank:
    def __init__(self, rn:float,):
        self.rn = rn # natural interest rate
        