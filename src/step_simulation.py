import numpy as np
import random
from utils import taxation, inflation, GDP, unemployment, init_agents, imbalance, gini_coefficient
from copy import deepcopy
from config import Configuration
from src.agent import agent
from src.firm import firm
from src.bank import bank   
from src.market import consumption



def step_simulation(config:Configuration, event:bool, intervention:bool, step:int, length:int, firm:firm, bank:bank, agents:list[agent], log:dict[int:dict]):
    '''
    simulate several steps of a simulation.
    '''
    random.seed(config.seed)
    np.random.seed(config.seed)
    
    unem_rate = log[step]['unemployment_rate']
    infla_rate = log[step]['inflation_rate']
    Nominal_GDP = log[step]['GDP']
    imba = log[step]['imbalance']
    
    for t in range(step+1, step+length+1):
        ########################## 事 件 开 始 ##########################
        # 降低就业意愿
        # 产量减少，工资收入减少
        # GDP下降
        
        # if event:
        #     if t >= 500 and t <= 900:
        #         work_state = []
        #         for a in agents:
        #             a.l = 1 if random.random() < 0.3 else 0
        #             work_state.append(a.l)
        
        # if event:
        #     if t == config.event_start:
        #         a_pw = [a.pw for a in agents]

        #     if t >= config.event_start and t <= config.event_end:
        #         for a, pw in zip(agents, a_pw):
        #             a.pw = (0.8 ** (1/(config.event_end-config.event_start))) ** (t-config.event_start) * pw # 在t=900时，就业意愿下降到t=500时的25%
        #     work_state = [a.work_decision() for a in agents] # work decision
        # else:
        #     work_state = [a.work_decision() for a in agents] # work decision
        work_state = [a.work_decision() for a in agents] # work decision
        
        if event and t in [100, 200, 300, 400]:
            firm.k_capital *= 1.05
            firm.k_labor = 1 - firm.k_capital
            print(f'{t}, k_labor: {firm.k_labor}, k_capital: {firm.k_capital}')
        ########################## 事 件 结 束 ##########################
        
        production = firm.produce(agents)               # 生产
        wages = firm.pay_wage(agents)                   # 支付工资
        taxes, wages_after_tax = taxation(wages)        # 计算个税
        
        for a, w in zip(agents, wages_after_tax):
            a.z = w # update monthly income
            bank.deposit(a.id, w + sum(taxes)/config.num_agents) # 再分配
        
        ########################## 干 预 开 始 ##########################
        if t >= config.intervent_start and t <= config.intervent_end and intervention:
            bank.natural_rate = max(bank.natural_rate * 1.002, 0.1)
        ########################## 干 预 结 束 ##########################
        
        if t % 3 == 0: imba = imbalance(agents, firm.P, firm.G, bank.deposits)
        
        ################################################
        # consumption in random order 随 机 顺 序 消 费 #
        ################################################
        total_money, total_quantity, deposits = consumption(config, agents, firm.G, firm.P, deepcopy(bank.deposits))
        firm.capital += total_money * (1-config.tax_rate_good)
        
        ######################################################
        # price and wage adjustment 调整工资, 价格, 投入资本量 #
        ######################################################
        firm.wage_adjustment(agents, imba)
        firm.price_adjustment(imba)
        firm.cap_adjustment(imba)
        firm.G -= total_quantity
        bank.deposits = deposits
        for a in agents:
            bank.deposit(a.id, total_money*config.tax_rate_good/config.num_agents) # 再分配
        
        ################################
        # annual operation 年 度 调 整 #
        ################################
        if t % 12 == 0:
            bank.interest(agents)                # interest payment
            unem_rate = unemployment(log)
            infla_rate = inflation(log)
            bank.rate_adjustment(unem_rate, infla_rate) 
            Nominal_GDP = GDP(log)
        
        log[t] = {
            'year': t // 12.1 + 1,
            'work_state': work_state, 
            'wage': [a.w for a in agents],
            'price': firm.P, 
            'rate': bank.rate, 
            'production': production,
            'imbalance':imba,
            'inflation_rate': infla_rate,
            'taxes': sum(taxes),
            'unemployment_rate': unem_rate,
            'deposit': deepcopy(bank.deposits),
            'avg_wage': sum([a.w for a in agents])/sum([1 if a.l else 0 for a in agents]),
            'GDP': Nominal_GDP,
            'capital': firm.capital,
            'assets': bank.assets,
            'gini': gini_coefficient([a.w for a in agents]),
            }
        
        if t % 6 == 0 and t > 30:
            for a in agents: a.adjust(t, log)
    print('finish', t)
    return firm, bank, agents, log

if __name__ == '__main__':
    from utils import plot_log, plot_bar
    config = Configuration()
    
    F = firm(A=config.A, 
             alpha_w=config.alpha_w, 
             alpha_p=config.alpha_p,
             alpha_c=config.alpha_c,
             init_good=config.init_good,
             init_cap=config.init_cap,
             k_labor=config.k_labor,
             k_capital=config.k_capital,
             )
    B = bank(rn=config.rn, 
             pi_t=config.pi_t, 
             un=config.un, 
             alpha_pi=config.alpha_pi, 
             alpha_u=config.alpha_u, 
             num_agents=config.num_agents, 
             rate_min=config.r_min,
             init_assets=config.init_assets,
             )
    agents = init_agents(config)
    
    F.P = np.mean([a.w*a.pc for a in agents]) # t=0 initial price
    unem_rate, infla_rate, Nominal_GDP, imba = 0., 0., 0., 0.
    log = {
        0:
            {
                'year': 0,
                'work_state': [a.l for a in agents], 
                'wage': [a.w for a in agents],
                'price': F.P, 
                'rate': B.rate, 
                'production': F.G,
                'imbalance': imba,
                'inflation_rate': infla_rate,
                'taxes': 0,
                'unemployment_rate': unem_rate,
                'deposit': {i: 0.0 for i in range(config.num_agents)},
                'avg_wage': sum([a.w for a in agents])/config.num_agents,
                'GDP': Nominal_GDP,
                'capital': F.capital,
                'assets': B.assets,
                'gini': gini_coefficient([a.w for a in agents]),
                }
            }
    F, B, agents, log = step_simulation(config, event=False, intervention=False, step=0, length=100, firm=deepcopy(F), bank=deepcopy(B), agents=deepcopy(agents), log=deepcopy(log))
    
    F, B, agents, log = step_simulation(config, event=False, intervention=False, step=100, length=200, firm=deepcopy(F), bank=deepcopy(B), agents=deepcopy(agents), log=deepcopy(log))
    
    plot_log('./figs/log_step.png', log, config)