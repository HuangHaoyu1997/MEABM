import numpy as np
import random
from utils import beta_dist, pay_wage, taxation, total_deposit, inflation, GDP
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
        
    
    def unemployment(self, log:dict):
        work_state_history = [log[key]['work_state'] for key in log.keys()]
        unemployment_cnt = 0
        tmp_states = work_state_history[-12:]
        for state in tmp_states:
            unemployment_cnt += state.count(0)
        return unemployment_cnt / (12 * len(state))
    
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
    
    

def simulation(config:Configuration):
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
        production = F.produce(agents) # production
        
        
        wages = pay_wage(agents)
        taxes = taxation(wages)
        # print(wages[:10], taxes[:10], '\n\n')
        wages_after_tax = [w - t for w, t in zip(wages, taxes)]
        for a, w in zip(agents, wages_after_tax):
            a.z = w # update monthly income
            # print(t, a.id, w, sum(taxes), w + sum(taxes)/config.num_agents)
            B.deposit(a.id, w + sum(taxes)/config.num_agents) # redistribution
        
        
        ###############################
        # consumption in random order #
        ###############################
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
        
        # for a in agents:
        #     a.consume_adjustment(F.P, B.deposits[a.id])
        #     # print(a.id, B.deposits[a.id])
        #     a.work_adjustment(B.deposits[a.id], B.rate)
        
        F.wage_adjustment(agents, phi_bar)
        F.price_adjustment(phi_bar)
        
        F.G = tmp_G
        # print(t, tmp_G)
        
        #####################
        # annual operation  #
        #####################
        if t % 12 == 0:
            B.interest(agents) # interest payment
            unem_rate = M.unemployment(log)
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
                sgn_price = 1 if this_period_price >= last_period_price else -1
                
                a.pw += sgn_deposit * np.random.uniform() * 0.3
                a.pw += sgn_wage * np.random.uniform() * 0.3
                a.pc += sgn_price * np.random.uniform() * 0.002
                a.pc += sgn_deposit * np.random.uniform() * 0.002

    return log

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from config import Configuration
    
    config = Configuration()
    
    log = simulation(config)
    # print(log[2])
    price_history = [log[key]['price'] for key in log.keys()]
    rate_history = [log[key]['rate'] for key in log.keys()]
    imba_history = [log[key]['imbalance'] for key in log.keys()]
    taxes_history = [log[key]['taxes']/config.num_agents for key in log.keys()]
    
    fig, axs = plt.subplots(3, 4, figsize=(24, 16))
    fig.suptitle('xxx')
    
    axs[0, 0].plot(price_history)
    axs[0, 0].set_xlabel('Time / Month'); axs[0, 0].set_ylabel('Price')
    axs[0, 0].grid()
    
    axs[0, 1].plot(rate_history)
    axs[0, 1].set_xlabel('Time / Month'); axs[0, 1].set_ylabel('Interest rate')
    axs[0, 1].grid()
    
    axs[0, 2].plot([log[key]['unemployment_rate'] for key in log.keys()])
    axs[0, 2].set_xlabel('Time / Month'); axs[0, 2].set_ylabel('Unemployment rate')
    axs[0, 2].grid()
    
    axs[0, 3].plot([log[key]['GDP'] for key in log.keys()])
    axs[0, 3].set_xlabel('Time / Month'); axs[0, 3].set_ylabel('Nominal GDP')
    axs[0, 3].grid()
    

    axs[1, 0].plot(imba_history)
    axs[1, 0].set_xlabel('Time / Month'); axs[1, 0].set_ylabel('Imbalance: Demand - Supply')
    axs[1, 0].grid()
    
    axs[1, 1].plot(taxes_history)
    axs[1, 1].set_xlabel('Time / Month'); axs[1, 1].set_ylabel('Avg tax revenue per capita')
    axs[1, 1].grid()
    
    axs[1, 2].plot([log[key]['production'] for key in log.keys()])
    axs[1, 2].set_xlabel('Time / Month'); axs[1, 2].set_ylabel('Production')
    axs[1, 2].grid()
    
    axs[2, 0].plot([total_deposit(log[key]['deposit'])/config.num_agents for key in log.keys()])
    axs[2, 0].set_xlabel('Time / Month'); axs[2, 0].set_ylabel('Deposit per capita')
    axs[2, 0].grid()
    
    axs[2, 1].plot([log[key]['avg_wage'] for key in log.keys()])
    axs[2, 1].set_xlabel('Time / Month'); axs[2, 1].set_ylabel('Avg wage')
    axs[2, 1].grid()
    
    axs[2, 2].plot([log[key]['inflation_rate'] for key in log.keys()])
    axs[2, 2].set_xlabel('Time / Month'); axs[2, 2].set_ylabel('Inflation rate')
    axs[2, 2].grid()
    
    # plt.tight_layout()

    plt.savefig('log.png', dpi=300)
    # plt.show()
    
    
    
    
    