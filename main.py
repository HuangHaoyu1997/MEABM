import numpy as np
import random
from utils import taxation, inflation, GDP, unemployment, init_agents, imbalance
from copy import deepcopy
from config import Configuration
from src.agent import agent
from src.firm import firm
from src.bank import bank   
    

def simulation(config:Configuration, event=False, intervention=False):
    '''one episode of simulation'''
    random.seed(config.seed)
    np.random.seed(config.seed)
    
    F = firm(A=config.A, 
             alpha_w=config.alpha_w, 
             alpha_p=config.alpha_p,
             init_good=config.init_good,
             init_cap=config.init_cap,
             )
    B = bank(rn=config.rn, 
             pi_t=config.pi_t, 
             un=config.un, 
             alpha_pi=config.alpha_pi, 
             alpha_u=config.alpha_u, 
             num_agents=config.num_agents, 
             rate_min=config.r_min,
             )
    agents = init_agents(config)

    F.P = np.mean([a.w for a in agents]) # t=0 initial price
    unem_rate, infla_rate, Nominal_GDP = 0., 0., 0.
    imba = 0.
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
                }
            }
    
    for t in range(1, config.num_time_steps+1):
        
        ################ 事 件 开 始 ################
        # 降低就业意愿
        # 产量减少，工资收入减少
        # GDP下降
        
        # if event:
        #     if t >= 500 and t <= 900:
        #         work_state = []
        #         for a in agents:
        #             a.l = 1 if random.random() < 0.3 else 0
        #             work_state.append(a.l)
        
        if event:
            if t == config.event_start:
                a_pw = [a.pw for a in agents]

            if t >= config.event_start and t <= config.event_end:
                for a, pw in zip(agents, a_pw):
                    # a.pw = 0.1 ** (1/300) * a.pw # 在t=900时，就业意愿下降到t=500时的25%
                    a.pw = (0.6 ** (1/(config.event_end-config.event_start))) ** (t-config.event_start) * pw
            work_state = [a.work_decision() for a in agents] # work decision
        else:
            work_state = [a.work_decision() for a in agents] # work decision
        ################ 事 件 结 束 ################
        
        production = F.produce(agents) # 生产
        wages = F.pay_wage(agents)     # 支付工资
        taxes = taxation(wages)        # 计算个税
        wages_after_tax = [w - t for w, t in zip(wages, taxes)]
        for a, w in zip(agents, wages_after_tax):
            a.z = w # update monthly income
            # print(t, a.id, w, sum(taxes), w + sum(taxes)/config.num_agents)
            
            ################ 干 预 开 始 ################
            if t >= config.intervent_start and t <= config.intervent_end and intervention:
                B.deposit(a.id, w + sum(taxes)/config.num_agents + 400) # 0.04*B.deposits[a.id] redistribution
            else:
                B.deposit(a.id, w + sum(taxes)/config.num_agents) # redistribution
            ################ 干 预 结 束 ################
        if t % 6 == 0:
            imba = imbalance(agents, F.P, F.G, B.deposits)
        
        ###########################################
        # consumption in random order 随机顺序消费 #
        ###########################################
        random_list = np.arange(config.num_agents)
        np.random.shuffle(random_list)
        tmp_G = deepcopy(F.G)
        total_consume_amount, total_consume_quantity = 0., 0.
        for idx in random_list: 
            a = agents[idx]
            demand = a.pc * B.deposits[a.id] / F.P # intended demand
            demand_actual = min(demand, tmp_G) # actual demand quanity
            consump_actual = demand_actual * F.P # actual consumption
            total_consume_quantity += demand_actual
            B.deposits[a.id] -= consump_actual # update deposit
            tmp_G -= demand_actual # update inventory of essential goods
            total_consume_amount += consump_actual
        F.capital += total_consume_amount
        # print(t, '总消费量: ', total_consume_quantity)
        #############################
        # price and wage adjustment #
        #############################
        
        F.wage_adjustment(agents, imba)
        F.price_adjustment(imba)
        
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
            'imbalance':imba,
            'inflation_rate': infla_rate,
            'taxes': sum(taxes),
            'unemployment_rate': unem_rate,
            'deposit': deepcopy(B.deposits),
            'avg_wage': sum([a.w for a in agents])/sum([1 if a.l else 0 for a in agents]),
            'GDP': Nominal_GDP,
            'capital': F.capital,
            }
        
        if t % 3 == 0 and t > 60:
            for a in agents:
                a.adjust(t, log)

    return log

if __name__ == '__main__':
    from utils import plot_log, plot_bar
    from config import Configuration
    from time import time
    
    config = Configuration()
    t1 = time()
    log = simulation(config)
    plot_log('./figs/log.png', log, config)
    print('running time:{:.3f}'.format(time()-t1))
    
    # config = Configuration()
    # config.seed = 123456
    # logs, logs_no_event = [], []
    # for i in range(5):
    #     print(f'Simulation {i+1}/5')
    #     config.seed += i
    #     log = simulation(config, event=True, intervention=True)
    #     logs.append(log)
    #     log = simulation(config, event=False, intervention=False)
    #     logs_no_event.append(log)
    # plot_bar('./figs/bar-event-intervention.png', logs, logs_no_event, config)
    
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