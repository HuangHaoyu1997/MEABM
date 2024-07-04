import numpy as np
import random
from utils import beta_dist, pay_wage
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
        self.rate = rn      # set initial interest rate to natural rate
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
        self.brackets = list(np.array([0, 9700, 39475, 84200, 160725, 204100, 510300])/12) # monthly income brackets
        self.rates = [0.1, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37] # tax rates
        
    def inflation(self, log:dict):
        '''
        通胀率 = 本年均价 - 上年均价 / 上年均价
        '''
        price_history = [log[key]['price'] for key in log.keys()]
        assert len(price_history) >= 12
        if len(price_history) < 12*2:
            return (np.mean(price_history[-12:]) - price_history[-12]) / price_history[-12]
        else:
            return (np.mean(price_history[-12:]) - np.mean(price_history[-12*2:-12])) /  np.mean(price_history[-12*2:-12])
    
    def unemployment(self, log:dict):
        work_state_history = [log[key]['work_state'] for key in log.keys()]
        unemployment_cnt = 0
        tmp_states = work_state_history[-12:]
        for state in tmp_states:
            unemployment_cnt += state.count(0)
        return unemployment_cnt / (12 * len(state))
    
    def taxation(self, wages:list[float]):
        taxes = []
        for w in wages:
            if w <= 0: taxes.append(0.); continue
            
            tax = 0.0
            for i in range(len(self.brackets) - 1):
                if w > self.brackets[i + 1]:
                    tax += (self.brackets[i + 1] - self.brackets[i]) * self.rates[i]
                else:
                    tax += (w - self.brackets[i]) * self.rates[i]
                    break
            if w > self.brackets[-1]:
                tax += (w - self.brackets[-1]) * self.rates[-1]
            taxes.append(tax)
        return taxes
    
    def total_intended_demand(self, agent_list:list[agent], P:float, deposits:dict) -> float:
        '''
        社会总预期需求 = 消费比例 * 储蓄量 / 价格
        '''
        cnt = np.sum([a.pc * deposits[a.id] / P for a in agent_list])
        return cnt
    
    def imbalance(self, agent_list:list[agent], P:float, G:float, deposits:dict) -> float:
        '''
        计算 预期需求与实际产量之间的不均衡
        '''
        D = self.total_intended_demand(agent_list, P, deposits)
        # print('imbalance:', D, firm.G)
        phi_bar = (D - G) / max(D, G)
        return phi_bar

def main(config:Configuration):
    
    M = market()
    F = firm(A=config.A, alpha_w=config.alpha_w, alpha_p=config.alpha_p)
    B = bank(rn=config.rn, pi_t=config.pi_t, un=config.un, alpha_pi=config.alpha_pi, alpha_u=config.alpha_u)
    
    agents = [agent(id=i, 
                    pw=np.random.uniform(0.6, 0.95), 
                    pc=np.random.uniform(0.05, 0.1), 
                    gamma=config.gamma, 
                    beta=config.gamma) for i in range(config.num_agents)]

    for a in agents:
        a.w = beta_dist()      # initial hourly wage
        B.deposits[a.id] = 0.  # initial deposit
    F.P = np.mean([a.w for a in agents]) # t=0 initial price
    unem_rate = 0.
    log = {}
    log[0] = {
        'work_state': [a.l for a in agents], 
        'price': F.P, 
        'rate': B.rate, 
        'production': 0,
        'imbalance': 0,
        'taxes': 0,
        'unemployment_rate': unem_rate,
        }
    
    for t in range(1, config.num_time_steps+1):
        
        work_state = [a.work_decision() for a in agents] # work decision
        production = F.produce(agents) # production
        
        
        wages = pay_wage(agents)
        taxes = M.taxation(wages)
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
            infla_rate = M.inflation(log)
            B.rate_adjustment(unem_rate, infla_rate) 
        
        log[t] = {
            'work_state': work_state, 
            'price': F.P, 
            'rate': B.rate, 
            'production': production,
            'imbalance':phi_bar,
            'taxes': sum(taxes),
            'unemployment_rate': unem_rate,
            }
        
        
    return log

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from config import Configuration
    
    config = Configuration()
    random.seed(config.seed)
    np.random.seed(config.seed)
    
    # Market = market()
    # taxes = Market.taxation([4500, 21000, 57000, 115000, 180000, 300000, 700000])
    # print(taxes)
    
    log = main(config)
    print(log[2])
    price_history = [log[key]['price'] for key in log.keys()]
    rate_history = [log[key]['rate'] for key in log.keys()]
    imba_history = [log[key]['imbalance'] for key in log.keys()]
    taxes_history = [log[key]['taxes']/config.num_agents for key in log.keys()]
    
    fig, axs = plt.subplots(2, 3, figsize=(16, 8))
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
    

    axs[1, 0].plot(imba_history)
    axs[1, 0].set_xlabel('Time / Month'); axs[1, 0].set_ylabel('Imbalance')
    axs[1, 0].grid()
    
    axs[1, 1].plot(taxes_history)
    axs[1, 1].set_xlabel('Time / Month'); axs[1, 1].set_ylabel('Avg tax revenue per capita')
    axs[1, 1].grid()
    
    # plt.tight_layout()

    plt.savefig('log.png', dpi=300)
    # plt.show()
    
    