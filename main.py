import numpy as np
import random
from utils import beta_dist, pay_wage, taxation, inflation, GDP, unemployment
from copy import deepcopy
from config import Configuration

class agent:
    def __init__(self, id:int, pw:float, pc:float, gamma:float, beta:float):
        self.id = id
        self.pw = pw         # probability of working
        self.w = 0           # hourly wage
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


class bank:
    def __init__(self, rn:float, pi_t:float, un:float, alpha_pi:float, alpha_u:float):
        self.rn = rn        # natural interest rate, constant value
        self.pi_t = pi_t    # target inflation rate, constant value
        self.un = un        # natural unemployment rate, constant value
        self.rate = rn      # initial interest rate = natural rate
        self.alpha_pi = alpha_pi
        self.alpha_u = alpha_u
        self.deposits = {}
    
    def interest(self, agent_list:list[agent],):
        '''
        interest rate payment anually
        '''
        for a in agent_list:
            self.deposits[a.id] *= (1 + self.rate)
    
    def deposit(self, agent_id, income:float):
        '''
        update the deposit of the agent with agent_id
        '''
        # print('before:', agent_id, self.deposits[agent_id])
        if agent_id in self.deposits:
            self.deposits[agent_id] += income
        else:
            self.deposits[agent_id] = income
        # print('after:', agent_id, self.deposits[agent_id])
    
    def rate_adjustment(self, unemployment_rate:float, inflation_rate:float):
        '''
        Taylor rule for interest rate adjustment
        
        '''
        rate_after = max(self.rn + self.pi_t + self.alpha_pi * (inflation_rate - self.pi_t) + self.alpha_u * (self.un - unemployment_rate), 0)
        self.rate = rate_after
        return rate_after

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
        production = sum([168 * self.A for a in agent_list if a.l==1])
        # for a in agent_list:
        #     if a.l == 1: production += 168 * self.A
        self.G += production
        return production
    
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
        return self.P
    
class market:
    def __init__(self, ) -> None:
        pass
    
    def total_intended_demand(self, agent_list:list[agent], P:float, deposits:dict) -> float:
        '''
        社会总预期需求 = 消费比例 * 储蓄量 / 价格
        '''
        cnt = np.sum([a.pc * deposits[a.id] / P for a in agent_list])
        return cnt
    
    def imbalance(self, agent_list:list[agent], P:float, G:float, deposits:dict) -> float:
        '''
        计算 预期需求与实际产量之间的不均衡
        >0, 需求 > 产量
        <0, 需求 < 产量
        '''
        D = self.total_intended_demand(agent_list, P, deposits)
        # print('imbalance:', D, firm.G)
        phi_bar = (D - G) / max(D, G)
        return phi_bar
    
    

def simulation(config:Configuration, event=False):
    '''
    one episode of simulation
    '''
    random.seed(config.seed)
    np.random.seed(config.seed)
    
    M = market()
    F = firm(A=config.A, alpha_w=config.alpha_w, alpha_p=config.alpha_p)
    B = bank(rn=config.rn, pi_t=config.pi_t, un=config.un, alpha_pi=config.alpha_pi, alpha_u=config.alpha_u)
    
    agents = [agent(id=i, 
                    pw=np.random.uniform(config.pw_low, config.pw_high), 
                    pc=np.random.uniform(config.pc_low, config.pc_high), 
                    gamma=config.gamma, 
                    beta=config.gamma) for i in range(config.num_agents)]

    for a in agents:
        a.w = beta_dist()      # initial hourly wage
        B.deposits[a.id] = 0.  # initial deposit
    F.P = np.mean([a.w for a in agents]) # t=0 initial price
    unem_rate = 0.
    infla_rate = 0.
    Nominal_GDP = 0.
    log = {}
    log[0] = {
        'year': 0,
        'work_state': [a.l for a in agents], 
        'wage': [a.w for a in agents],
        'price': F.P, 
        'rate': B.rate, 
        'production': 0,
        'imbalance': 0,
        'inflation_rate': infla_rate,
        'taxes': 0,
        'unemployment_rate': unem_rate,
        'deposit': {i: 0.0 for i in range(config.num_agents)},
        'avg_wage': sum([a.w for a in agents])/config.num_agents,
        'GDP': Nominal_GDP,
        }
    
    for t in range(1, config.num_time_steps+1):
        
        work_state = [a.work_decision() for a in agents] # work decision
        
        ################ 干 预 ################
        # 增加失业
        # 产量减少
        # GDP下降
        
        if event:
            if t >= 500 and t <= 900:
                work_state = []
                for a in agents:
                    a.l = 1 if random.random() < 0.3 else 0
                    work_state.append(a.l)
        
        production = F.produce(agents) # production
        
        
        wages = pay_wage(agents)
        taxes = taxation(wages)
        # print(wages[:10], taxes[:10], '\n\n')
        wages_after_tax = [w - t for w, t in zip(wages, taxes)]
        for a, w in zip(agents, wages_after_tax):
            a.z = w # update monthly income
            # print(t, a.id, w, sum(taxes), w + sum(taxes)/config.num_agents)
            
            if t >= 550 and t <= 900 and event and False:
                B.deposit(a.id, w + sum(taxes)/config.num_agents + 0.04*B.deposits[a.id]) # redistribution
            else:
                B.deposit(a.id, w + sum(taxes)/config.num_agents) # redistribution
        
        
        ###########################################
        # consumption in random order 随机顺序消费 #
        ###########################################
        random_list = np.arange(config.num_agents)
        np.random.shuffle(random_list)
        tmp_G = deepcopy(F.G)
        for idx in random_list: 
            a = agents[idx]
            demand = a.pc * B.deposits[a.id] / F.P # intended demand
            demand_actual = min(demand, tmp_G) # actual demand quanity
            consump_actual = demand_actual * F.P # actual consumption
            B.deposits[a.id] -= consump_actual # update deposit
            tmp_G -= demand_actual # update inventory of essential goods
        
        #############################
        # price and wage adjustment #
        #############################
        phi_bar = M.imbalance(agents, F.P, F.G, B.deposits)
        F.wage_adjustment(agents, phi_bar)
        F.price_adjustment(phi_bar)
        
        F.G = tmp_G
        
        #####################
        # annual operation  #
        #####################
        if t % 12 == 0:
            B.interest(agents) # interest payment
            unem_rate = unemployment(log)
            infla_rate = inflation(log)
            B.rate_adjustment(unem_rate, infla_rate) 
            Nominal_GDP = GDP(log)
        
        
        log[t] = {
            'year': t // 12.1 + 1,
            'work_state': work_state, 
            'wage': [a.w for a in agents],
            'price': F.P, 
            'rate': B.rate, 
            'production': production,
            'imbalance':phi_bar,
            'inflation_rate': infla_rate,
            'taxes': sum(taxes),
            'unemployment_rate': unem_rate,
            'deposit': deepcopy(B.deposits),
            'avg_wage': sum([a.w for a in agents])/config.num_agents,
            'GDP': Nominal_GDP,
            }
        
        if t % 6 == 0 and t > 60:
            for a in agents:
                last_period_deposit = sum([log[tt]['deposit'][a.id]*(1+log[tt]['rate']) for tt in range(t-12+1, t-6+1)])
                this_period_deposit = sum([log[tt]['deposit'][a.id]*(1+log[tt]['rate']) for tt in range(t-6+1, t+1)])
                sgn_deposit = -1 if this_period_deposit >= last_period_deposit else 1
                
                last_period_wage = sum([log[tt]['wage'][a.id] for tt in range(t-12+1, t-6+1)])
                this_period_wage = sum([log[tt]['wage'][a.id] for tt in range(t-6+1, t+1)])
                sgn_wage = 1 if this_period_wage >= last_period_wage else -1
                
                last_period_price = sum([log[tt]['price'] for tt in range(t-12+1, t-6+1)])
                this_period_price = sum([log[tt]['price'] for tt in range(t-6+1, t+1)])
                sgn_price = -1 if this_period_price >= last_period_price else 1
                
                a.pw += sgn_deposit * np.random.uniform() * config.pw_delta # 存款多了减少工作意愿
                a.pw += sgn_wage * np.random.uniform() * config.pw_delta    # 工资多了增加工作意愿
                a.pc += sgn_price * np.random.uniform() * config.pc_delta   # 价格上涨减少消费意愿
                a.pc += sgn_deposit * np.random.uniform() * config.pc_delta # 存款多了增加消费意愿

    return log

if __name__ == '__main__':
    from utils import plot_log, plot_bar
    from config import Configuration
    
    config = Configuration()
    config.seed = 123456
    logs = []
    for i in range(5):
        print(f'Simulation {i+1}/5')
        config.seed += i
        log = simulation(config, event=True)
        logs.append(log)
    
    
    config.seed = 123456
    logs_no_event = []
    for i in range(5):
        print(f'Simulation {i+1}/5')
        config.seed += i
        log = simulation(config, event=False)
        logs_no_event.append(log)
    plot_bar('bar-event.png', logs, logs_no_event, config)
    # log = simulation(config)
    # plot_log('log.png', log, config)