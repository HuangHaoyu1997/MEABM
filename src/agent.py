import random
import numpy as np

def gauss_dist(mean_income, std_dev_income, num_samples=100):
    return np.random.normal(mean_income, std_dev_income, num_samples)

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
    def __init__(self, 
                 id:int, 
                 pw:float, 
                 pc:float, 
                 gamma:float, 
                 beta:float, 
                 pw_delta:float, 
                 pc_delta:float,
                 ):
        self.id = id
        self.pw = pw         # probability of working
        self.w = beta_dist() # hourly wage
        self.z = 0           # monthly income
        self.pc = pc         # proportion of consumption
        self.l = 0           # work state (0: unemployed, 1: employed)
        self.gamma = gamma
        self.beta = beta
        self.pw_delta = pw_delta
        self.pc_delta = pc_delta
        
        
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
    
    def adjust(self, timestep:int, log:dict):
        '''
        维持消费品的消费量稳定
        因此
        价格上升，消费比例会提升
        存款增加，消费比例会降低
        '''
        
        
        last_period_deposit = sum([log[tt]['deposit'][self.id]*(1+log[tt]['rate']) for tt in range(timestep-6+1, timestep-3+1)])
        this_period_deposit = sum([log[tt]['deposit'][self.id]*(1+log[tt]['rate']) for tt in range(timestep-3+1, timestep+1)])
        sgn_deposit = -1 if this_period_deposit >= last_period_deposit else 1
        
        last_period_wage = sum([log[tt]['wage'][self.id] for tt in range(timestep-6+1, timestep-3+1)])
        this_period_wage = sum([log[tt]['wage'][self.id] for tt in range(timestep-3+1, timestep+1)])
        sgn_wage = 1 if this_period_wage >= last_period_wage else -1
        
        last_period_price = sum([log[tt]['price'] for tt in range(timestep-6+1, timestep-3+1)])
        this_period_price = sum([log[tt]['price'] for tt in range(timestep-3+1, timestep+1)])
        sgn_price = 1 if this_period_price >= last_period_price else -1
        
        self.pw += sgn_deposit * np.random.uniform() * self.pw_delta # 存款多了减少工作意愿
        self.pw += sgn_wage * np.random.uniform() * self.pw_delta    # 工资多了增加工作意愿
        self.pw = max(0, min(self.pw, 1.0))
        
        self.pc += sgn_price * np.random.uniform() * self.pc_delta   # 价格上涨增加消费比例（维持消费量不变）
        self.pc += sgn_deposit * np.random.uniform() * self.pc_delta # 存款多了降低消费比例（维持消费量不变）
        self.pc = max(0, min(self.pc, 1.0))
        
        if timestep % 12 in [12, 1, 2]: # 冬季降低就业意愿
            self.pw *= 0.95