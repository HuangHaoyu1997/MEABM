import numpy as np
import random
from utils import taxation, inflation, GDP, unemployment, init_agents, imbalance, gini_coefficient
from copy import deepcopy
from config import Configuration
from src.agent import agent
from src.firm import firm
from src.bank import bank   
from src.market import consumption


def simulation(config:Configuration, event=False, intervention=False):
    '''one episode of simulation'''
    random.seed(config.seed)
    np.random.seed(config.seed)
    a_w = []
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
    
    for t in range(1, config.num_time_steps+1):
        
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
        
        
        work_state = [a.work_decision() for a in agents] # work decision
        
        ########### 实验三: 信息革命
        # if event and t in [100, 200, 300, 400]:
        #     F.k_capital *= 1.05
        #     F.k_labor = 1 - F.k_capital
        #     print(f'{t}, k_labor: {F.k_labor}, k_capital: {F.k_capital}')
            
        ########### 实验二: 战后重建
        if event:
            if t == config.event_start:
                a_pw = [a.pw for a in agents]

            if t >= config.event_start and t <= config.event_end:
                for a, pw in zip(agents, a_pw):
                    a.pw = (2.0 ** (1/(config.event_end-config.event_start))) ** (t-config.event_start) * pw # 在t=900时，就业意愿下降到t=500时的25%
            work_state = [a.work_decision() for a in agents] # work decision
        else:
            work_state = [a.work_decision() for a in agents] # work decision
            
        ########### 实验一: 经济危机
        if event:
            if t == config.event_start:
                a_pw = [a.pw for a in agents]

            if t >= config.event_start and t <= config.event_end:
                for a, pw in zip(agents, a_pw):
                    a.pw = (0.9 ** (1/(config.event_end-config.event_start))) ** (t-config.event_start) * pw # 在t=900时，就业意愿下降到t=500时的25%
            work_state = [a.work_decision() for a in agents] # work decision
        else:
            work_state = [a.work_decision() for a in agents] # work decision
        ########################## 事 件 结 束 ##########################
        
        production = F.produce(agents) # 生产
        wages = F.pay_wage(agents)     # 支付工资
        taxes, wages_after_tax = taxation(wages)        # 计算个税
        
        for a, w in zip(agents, wages_after_tax):
            a.z = w # update monthly income
            B.deposit(a.id, w + sum(taxes)/config.num_agents) # 再分配
            
            
            ########################## 干 预 开 始 ##########################
            if t >= config.intervent_start and t <= config.intervent_end and intervention:
                B.deposit(a.id, w + sum(taxes)/config.num_agents + 200) # 0.04*B.deposits[a.id] redistribution
            else:
                B.deposit(a.id, w + sum(taxes)/config.num_agents)
            ########################## 干 预 结 束 ##########################
        
        
        ########################## 干 预 开 始 ##########################
        if t >= config.intervent_start and t <= config.intervent_end and intervention:
            B.natural_rate = max(B.natural_rate * 1.002, 0.1)
        ########################## 干 预 结 束 ##########################
        
        
        if t % 3 == 0:
            imba = imbalance(agents, F.P, F.G, B.deposits)
        
        ################################################
        # consumption in random order 随 机 顺 序 消 费 #
        ################################################
        total_money, total_quantity, deposits = consumption(config, agents, F.G, F.P, deepcopy(B.deposits))
        F.capital += total_money * (1-config.tax_rate_good)
        # print(t, total_money, total_money*config.tax_rate_good)
        
        # print(t, '总消费量: ', total_money)
        
        ######################################################
        # price and wage adjustment 调整工资, 价格, 投入资本量 #
        ######################################################
        F.wage_adjustment(agents, imba)
        F.price_adjustment(imba)
        F.cap_adjustment(imba)
        F.G -= total_quantity
        B.deposits = deposits
        for a in agents:
            B.deposit(a.id, total_money*config.tax_rate_good/config.num_agents) # 再分配
        
        ############################
        # annual operation 年度调整 #
        ############################
        if t % 12 == 0:
            B.interest(agents)                # interest payment
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
            'imbalance':imba,
            'inflation_rate': infla_rate,
            'taxes': sum(taxes),
            'unemployment_rate': unem_rate,
            'deposit': deepcopy(B.deposits),
            'avg_wage': sum([a.w for a in agents])/sum([1 if a.l else 0 for a in agents]),
            'GDP': Nominal_GDP,
            'capital': F.capital,
            'assets': B.assets,
            'gini': gini_coefficient([B.deposits[a.id] for a in agents]),
            }
        
        if t % 6 == 0 and t > 30:
            for a in agents:
                a.adjust(t, log)
        a_w.append(agents[0].pc)
    # import matplotlib.pyplot as plt
    # plt.plot(a_w)
        # print(t, F.cap4product)
    # plt.show()

    return log

if __name__ == '__main__':
    from utils import plot_log, plot_bar
    from config import Configuration
    from time import time
    ####################################### 单 次 实 验 #######################################
    # config = Configuration()
    # t1 = time()
    # log = simulation(config, event=True, intervention=True)
    # plot_log('./figs/log.png', log, config)
    # print('running time:{:.3f}'.format(time()-t1))
    
    ####################################### 多 次 实 验 #######################################
    # config = Configuration()
    # config.seed = 123456
    # logs = []
    # for i in range(5):
    #     print(f'Simulation {i+1}/5')
    #     config.seed += i
    #     log = simulation(config, event=False, intervention=True)
    #     logs.append(log)
    # plot_bar('./figs/bar-no-event-no-intervention.png', logs, logs_compare=None, config=config)
    
    
    ####################################### 对 照 试 验 #######################################
    config = Configuration()
    config.seed = 123456
    logs, logs_no_event = [], []
    for i in range(5):
        print(f'Simulation {i+1}/5')
        config.seed += i
        log = simulation(config, event=True, intervention=True)
        logs.append(log)
        log = simulation(config, event=False, intervention=False)
        logs_no_event.append(log)
    plot_bar('./figs/bar-event-intervention.png', logs, logs_no_event, config)
    
    # config.seed = 123456
    # logs = []
    # for i in range(5):
    #     print(f'Simulation {i+1}/5')
    #     config.seed += i
    #     log = simulation(config, event=True, intervention=False)
    #     logs.append(log)
    # plot_bar('./figs/bar-event-no-intervention.png', logs, logs_no_event, config)
    
    # config.seed = 123456
    # logs = []
    # for i in range(5):
    #     print(f'Simulation {i+1}/5')
    #     config.seed += i
    #     log = simulation(config, event=False, intervention=True)
    #     logs.append(log)
    # plot_bar('./figs/bar-no-event-intervention.png', logs, logs_no_event, config)
    # 
    # 