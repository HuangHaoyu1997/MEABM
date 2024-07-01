import numpy as np
import random
from utils import beta_dist

class agent:
    def __init__(self, id:int, pw:float, pc:float, gamma:float, beta:float):
        self.id = id
        self.pw = pw # probability of working
        self.w = 0 # wage
        self.z = 0 # monthly income
        self.pc = pc # proportion of consumption
        self.l = None
        self.gamma = gamma
        self.beta = beta
        
    def work_decision(self,):
        l = 1 if random.random() < self.pw else 0
        self.l = l
        return l
    
    def work_adjustment(self, deposit:float, rate:float):
        self.pw = pow((self.z + 1e-5) / (deposit + deposit*rate + 1e-5), self.gamma)
        print(self.id, "pw:", self.pw)
    
    def consume_adjustment(self, P:float, deposit:float):
        self.pc = pow(P / (deposit + self.z + 1e-5), self.beta)
class bank:
    def __init__(self, rn:float, pi_t:float, un:float, alpha_pi:float, alpha_u:float):
        self.rn = rn # natural interest rate
        self.pi_t = pi_t # target inflation rate
        self.un = un # natural unemployment rate
        self.rate = rn
        self.alpha_pi = alpha_pi
        self.alpha_u = alpha_u
        self.deposits = {}
    
    def interest(self, agent_list:list[agent],):
        for a in agent_list:
            self.deposits[a.id] *= (1 + self.rate)
    
    def deposit(self, agent_id, income:float):
        if agent_id in self.deposits:
            self.deposits[agent_id] += income
        else:
            self.deposits[agent_id] = income
    
    def rate_adjustment(self, unemployment_rate:float, inflation_rate:float):
        self.rate = max(self.rn + self.pi_t + self.alpha_pi * (inflation_rate - self.pi_t) + self.alpha_u * (self.un - unemployment_rate), 0)
    
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
    
    def pay_wage(self, agent_list:list[agent],):
        wages = []
        for a in agent_list:
            if a.l:
                a.z = a.w * 168 # monthly income
                wages.append(a.z)
        return wages
    
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
        self.brackets = list(np.array([0, 9700, 39475, 84200, 160725, 204100, 510300])/12) # monthly income brackets
        self.rates = [0.1, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37] # tax rates
        
    def inflation(self, price_history:list[float]):
        '''
        通胀率 = 本年均价 - 上年均价 / 上年均价
        '''
        assert len(price_history) >= 12
        if len(price_history) < 12*2:
            return (np.mean(price_history[-12:]) - price_history[-12]) / price_history[-12]
        else:
            return (np.mean(price_history[-12:]) - np.mean(price_history[-12*2:-12])) /  np.mean(price_history[-12*2:-12])
    
    def unemployment(self, work_state_history:list[list[int]]):
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
    
    def total_intended_demand(self, agent_list:list[agent], firm:firm, bank:bank) -> float:
        '''
        社会总预期需求 = 消费比例 * 储蓄量 / 价格
        '''
        cnt = np.sum([a.pc * bank.deposits[a.id] / firm.P for a in agent_list])
        return cnt
    
    def imbalance(self, agent_list:list[agent], firm:firm, bank:bank) -> float:
        '''
        计算 预期需求与实际产量之间的不均衡
        '''
        D = self.total_intended_demand(agent_list, firm, bank)
        phi_bar = (D - firm.G) / max(D, firm.G)
        return phi_bar

def main():
    N = 100
    T = 12*20 # 20 years
    
    M = market()
    F = firm(A=1, alpha_w=0.05, alpha_p=0.1)
    B = bank(rn=0.01, pi_t=0.02, un=0.04, alpha_pi=0.5, alpha_u=0.5)
    agents = [agent(id=i, pw=np.random.uniform(0.3, 0.7), pc=np.random.uniform(0.2, 0.4), gamma=0.5, beta=0.5) for i in range(N)]

    for a in agents:
        a.w = beta_dist()
        B.deposits[a.id] = 0.
    
    
    
    work_state_history = []
    price_history = []
    rate_history = []
    F.P = np.mean([a.w for a in agents]) # t=0 initial price
    price_history.append(F.P)
    
    for t in range(1, T+1):
        work_state = [a.work_decision() for a in agents] # work decision
        work_state_history.append(work_state) # record work state
        rate_history.append(B.rate) # record interest rate
        F.produce(agents) # production
        wages = F.pay_wage(agents)
        taxes = M.taxation(wages)
        wages_after_tax = [w - t for w, t in zip(wages, taxes)]
        for a, w in zip(agents, wages_after_tax):
            a.z = w # update monthly income
            B.deposit(a.id, w + sum(taxes)/N) # redistribution
        
        
        ######################
        # random consumption #
        ######################
        random_list = np.arange(N)
        np.random.shuffle(random_list)
        for idx in random_list: 
            a = agents[idx]
            demand = a.pc * B.deposits[a.id] / F.P # intended demand
            demand_actual = min(demand, F.G) # actual demand quanity
            consump_actual = demand_actual * F.P # actual consumption
            B.deposits[a.id] -= consump_actual # update deposit
            F.G -= demand_actual # update inventory of essential goods
        
        #############################
        # price and wage adjustment #
        #############################
        phi_bar = M.imbalance(agents, F, B)
        F.wage_adjustment(agents, phi_bar)
        F.price_adjustment(phi_bar)
        
        for a in agents:
            a.consume_adjustment(F.P, B.deposits[a.id])
            a.work_adjustment(B.deposits[a.id], B.rate)
        
        price_history.append(F.P)
        
        #####################
        # annual operation  #
        #####################
        if t % 12 == 0:
            B.interest(agents) # interest payment
            unem_rate = M.unemployment(work_state_history)
            infla_rate = M.inflation(price_history)
            B.rate_adjustment(unem_rate, infla_rate) 
    return price_history, rate_history

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    price_history, rate_history = main()
    
    plt.subplot(121)
    plt.plot(price_history)
    plt.xlabel('Time/Month')
    plt.ylabel('Price')
    
    
    plt.subplot(122)
    plt.plot(rate_history)
    plt.xlabel('Time/Month')
    plt.ylabel('Interest rate')

    
    # plt.title('Price history')
    plt.savefig('price_history.png')
    plt.show()
    
    