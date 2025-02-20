import sys
import os
current_dir = os.path.dirname(__file__) # 获取当前文件的目录
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir)) # 获取上级目录
sys.path.append(parent_dir) # 将上级目录添加到 sys.path

import numpy as np
import random
from copy import deepcopy
from config import Configuration, EconomicCrisisConfig, ReconstructionConfig, EconomicProsperityConfig
from src.agent import agent
from src.firm import firm
from src.bank import bank   
from src.market import consumption, taxation
from src.utils import inflation, GDP, unemployment, init_agents, imbalance, gini_coefficient, generate_unique_pairs
from src.opinion_dynamics import Deffuant_Weisbuch

def step_simulation(
        config:Configuration|EconomicCrisisConfig|ReconstructionConfig|EconomicProsperityConfig, 
        event:int, 
        intervention:bool, 
        action: float,
        step:int, 
        length:int, 
        firm:firm, 
        bank:bank, 
        agents:list[agent], 
        log:dict[int:dict],
        ):
    
    '''
    simulate several steps of a simulation.
    config: Configuration file
    event: int, event id2
    intervention: bool, whether to apply intervention
    step: int, current step
    length: int, length of simulation steps
    firm: firm object
    bank: bank object
    agents: list of agent objects
    log: dict of logs


    '''
    # random.seed(config.seed)
    # np.random.seed(config.seed)
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
        
        #                   .@@@.               ............ .   . .@@@ ..                                
        #       .............@@@@............   .@@@@@@@@@@. . .  .@@@@.. .            ..... .            
        #     . .@@@@@@@@@@@@@@@@@@@@@@@@@@@. .  ........@@. .  .@@@..@@@.       ....@@@@@@. .            
        #     . .@@@.....................@@@. ....@@..  .@@.  ..@@@.  ..@@@.    ....@@@@@@@. .            
        #     . .@@@. .@@..   .@@.     . @@@. ....@@.. .@@@ ..@@@..     .@@@@...    .. .@@@. .            
        #       ..... ..@@@@. @@@. .   . ....  . .@@.. .@@@.@@@.@@@@@@@@@@.@@@.. .   . .@@@. .            
        #          ...   .@@. @@@. .          .. .@@.  .@@..... ........... ... .    . .@@@. .            
        #       ...@@@@..    .@@@. .          ....@@....@@... ...  ....    ...       . .@@@. .            
        #          ..@@@@. . .@@@. .          . ..@@@@@@@@@@@ .@@. .@@@ . .@@....    . .@@@. .            
        #      ...............@@..............  ..........@@. .@@.. .@@...@@@. .     . .@@@. .            
        #    ....@@@@@@@@@@@@@@@@@@@@@@@@@@@@. .    ......@@.  .@@. .@@. .@@. .      . .@@@. .            
        #    ...............@@@............... ....@@@@@..@@.  .@@. ..@..@@. ..      . .@@@. .            
        #                .@@@@..@@@@@..      . .@@@@......@@.   ...  ...@@.       ......@@@......         
        #           ....@@@@.. ....@@@@@@...   ...      .@@@ ..........@@..........@@@@@@@@@@@@@. .       
        #       ...@@@@@@@.          ..@@@@@.      .....@@@..@@@@@@@@@@@@@@@@@. .................  
        if event == 1:
            # if t == config.event_start:
            #     a_pw = [a.pw for a in agents]

            if t > config.event_start and t <= config.event_end:
                # print(t, log.keys())
                for a, pw in zip(agents, log[config.event_start]['pw']):
                    a.pw = (config.decay_rate ** (1/(config.event_end-config.event_start))) ** (t-config.event_start) * pw # 在t=900时，就业意愿下降到t=500时的25%
            work_state = [a.work_decision() for a in agents] # work decision


        #                   .@@@.               ............ .   . .@@@ ..                                
        #       .............@@@@............   .@@@@@@@@@@. . .  .@@@@.. .         ....@....             
        #     . .@@@@@@@@@@@@@@@@@@@@@@@@@@@. .  ........@@. .  .@@@..@@@.     ....@@@@@@@@@@...          
        #     . .@@@.....................@@@. ....@@..  .@@.  ..@@@.  ..@@@.    ..@@.......@@@....        
        #     . .@@@. .@@..   .@@.     . @@@. ....@@.. .@@@ ..@@@..     .@@@@.......       .@@@...        
        #       ..... ..@@@@. @@@. .   . ....  . .@@.. .@@@.@@@.@@@@@@@@@@.@@@.. .       . .@@@....       
        #          ...   .@@. @@@. .          .. .@@.  .@@..... ........... ... .          .@@@. .        
        #       ...@@@@..    .@@@. .          ....@@....@@... ...  ....    ...          . .@@@...         
        #          ..@@@@. . .@@@. .          . ..@@@@@@@@@@@ .@@. .@@@ . .@@....     .  .@@@. .          
        #      ...............@@..............  ..........@@. .@@.. .@@...@@@. .       ..@@@. .           
        #    ....@@@@@@@@@@@@@@@@@@@@@@@@@@@@. .    ......@@.  .@@. .@@. .@@. .      ..@@@@. .            
        #    ...............@@@............... ....@@@@@..@@.  .@@. ..@..@@. ..     .@@@@.                
        #                .@@@@..@@@@@..      . .@@@@......@@.   ...  ...@@.      ..@@@@..........         
        #           ....@@@@.. ....@@@@@@...   ...      .@@@ ..........@@...... ..@@@@@@@@@@@@@@.. .      
        #       ...@@@@@@@.          ..@@@@@.      .....@@@..@@@@@@@@@@@@@@@@@. ..................            
        if event == 2:
            if t == config.event_start:
                a_pw = [a.pw for a in agents]

            if t >= config.event_start and t <= config.event_end:
                for a, pw in zip(agents, a_pw):
                    a.pw = (2.0 ** (1/(config.event_end-config.event_start))) ** (t-config.event_start) * pw # 在t=900时，就业意愿下降到t=500时的25%
            work_state = [a.work_decision() for a in agents] # work decision


        #                   .@@@.               ............ .   . .@@@ ..                   .            
        #       .............@@@@............   .@@@@@@@@@@. . .  .@@@@.. .         ....@....             
        #     . .@@@@@@@@@@@@@@@@@@@@@@@@@@@. .  ........@@. .  .@@@..@@@.       ..@@@@@@@@@@@...         
        #     . .@@@.....................@@@. ....@@..  .@@.  ..@@@.  ..@@@.   . .@@.......@@@@...        
        #     . .@@@. .@@..   .@@.     . @@@. ....@@.. .@@@ ..@@@..     .@@@@...   .       .@@@....       
        #       ..... ..@@@@. @@@. .   . ....  . .@@.. .@@@.@@@.@@@@@@@@@@.@@@....         .@@@....       
        #          ...   .@@. @@@. .          .. .@@.  .@@..... ........... ... .    ......@@@....        
        #       ...@@@@..    .@@@. .          ....@@....@@... ...  ....    ...     . .@@@@@@.. .          
        #          ..@@@@. . .@@@. .          . ..@@@@@@@@@@@ .@@. .@@@ . .@@....  . ..@@@@@@@.           
        #      ...............@@..............  ..........@@. .@@.. .@@...@@@. .         ...@@@....       
        #    ....@@@@@@@@@@@@@@@@@@@@@@@@@@@@. .    ......@@.  .@@. .@@. .@@. .             .@@@.         
        #    ...............@@@............... ....@@@@@..@@.  .@@. ..@..@@. ..   .         .@@@.         
        #                .@@@@..@@@@@..      . .@@@@......@@.   ...  ...@@.     .@@@..    ..@@@@. .       
        #           ....@@@@.. ....@@@@@@...   ...      .@@@ ..........@@...... ..@@@@@@@@@@@@..          
        #       ...@@@@@@@.          ..@@@@@.      .....@@@..@@@@@@@@@@@@@@@@@. .  ..@@@@@@... 
        if event == 3 and t in [100, 200, 300, 400]:
            firm.k_capital *= 1.05
            firm.k_labor = 1 - firm.k_capital
            print(f'{t}, k_labor: {firm.k_labor}, k_capital: {firm.k_capital}')
        
        ########################## 事 件 结 束 ##########################
        
        production = firm.produce(agents, heterogeneity=False)               # 生产
        wages = firm.pay_wage(agents)                   # 支付工资
        taxes, wages_after_tax = taxation(wages)        # 计算个税
        
        for a, w in zip(agents, wages_after_tax):
            a.z = w # update monthly income
            # print('before: ', w + sum(taxes)/config.num_agents)
            bank.deposit(a.id, w + sum(taxes)/config.num_agents) # 再分配
            
            ########################## 干 预 开 始 ##########################
            # if t >= config.intervent_start and t <= config.intervent_end and intervention:
            #     bank.deposit(a.id, w + sum(taxes)/config.num_agents + 200) # 0.04*B.deposits[a.id] redistribution
            # else:
            #     bank.deposit(a.id, w + sum(taxes)/config.num_agents)
            ########################## 干 预 结 束 ##########################
        
        ########################## 干 预 开 始 ##########################
        # if t >= config.intervent_start and t <= config.intervent_end and intervention:
        #     bank.natural_rate = max(bank.natural_rate * 1.002, 0.1)
        if intervention:
            bank.rate = float(action)
        ########################## 干 预 结 束 ##########################
        
        if t % 3 == 0:
            imba = imbalance(agents, firm.P, firm.G, bank.deposits)
        
        ################################################
        # consumption in random order 随 机 顺 序 消 费 #
        ################################################
        total_money, total_quantity, deposits = consumption(config, agents, firm.G, firm.P, deepcopy(bank.deposits), firm.rng)
        firm.capital += total_money * (1-config.tax_rate_good)
        # print(t, total_money, total_money*config.tax_rate_good)
        
        # print(t, '总消费量: ', total_money)
        
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
            # bank.rate_adjustment(unem_rate, infla_rate) 
            Nominal_GDP = GDP(log)
        
        log[t] = {
            'year': t // 12.1 + 1,
            'work_state': work_state, 
            'wage': [a.w for a in agents],
            'pw': [a.pw for a in agents],
            'pc': [a.pc for a in agents],
            'price': firm.P, 
            'rate': bank.rate, 
            'inventory': firm.G,
            'production': production,
            'imbalance':imba,
            'inflation_rate': infla_rate,
            'taxes': sum(taxes),
            'unemployment_rate': 1-sum([a.l for a in agents])/config.num_agents,
            'deposit': deepcopy(bank.deposits),
            'avg_wage': sum([a.w for a in agents])/sum([1 if a.l else 0 for a in agents]) if sum([1 if a.l else 0 for a in agents]) else 0.0,
            'GDP': Nominal_GDP,
            'capital': firm.capital,
            'assets': bank.assets,
            'gini': gini_coefficient([a.w for a in agents]),
            }
        
        #   ..........@@@@@@@@@. .         .@@.........  .@@@@@@@@@..  .@.            .@@@@..             
        #   ......@@..@.......@@ .         .@@@@@@@@@@.  ...............@....       ..@@...@@..           
        #   ...   .@..@. .@. .@@ .         .@@ .    ..  . @@.....@@ .@@@@@@@@..   .@@@.    ..@@..         
        #   .@@...@@ .@. .@. .@@ . . .@@@@@@@@@@@@@@. .   .@@@@@@@@ @@..@..@@ ...@@@@.........@@@@..      
        #     .@@.@. .@. .@. .@@ . . @@...........@@. .  ...........@@..@. @@ ..@.............. ..@.      
        #      .@@@. .@. .@. .@@ . . @@.          .@. .. .@.@@.@@.@..@.@@..@@ .   ...............         
        #      .@@@. .@..@@@..@.   . .@@@@@@@@@@@@@@. .. .@.@. @@.@...@@@@...     .@@@@@@@@@@@@@.         
        #     .@@.@@.  .@@@@.  ... . ................  . .@.@.@@@.@@   .@..@.   . .@.       . @@.         
        #   ..@@. .@...@@..@.  .@.  .@@. .@.  @@. .@@. . .@...@...@@   .@..@@.  . .@.         @@.         
        #   .@@.   ..@@@. .@@..@@. .@@.  .@.  .@@. .@@.  .@. .@...@@.@@@@@@@@.  . .@@@@@@@@@@@@@.
        
        # unique_pairs = generate_unique_pairs(config.num_agents, config.communications_num) # 总可能数(M-2)(M+1)/2≈5000
        # for pair in unique_pairs:
        #     # print('beforeDeffuant_Weisbuch:', agents[pair[0]].pw, agents[pair[1]].pw)
        #     update_flag = (agents[pair[0]], agents[pair[1]], 'w', config.bounded_conf, config.opinion_fusion_factor)
        #     # if update_flag: print('after:', agents[pair[0]].pw, agents[pair[1]].pw, '\n')

        if t % 6 == 0 and t > 30:
            for a in agents: a.adjust(t, log)
            
    # print('finish at timestep: ', t)
    return firm, bank, agents, log

if __name__ == '__main__':
    from utils import plot_log, plot_bar
    config = Configuration()
    
    F0 = firm(A=config.A, 
             alpha_w=config.alpha_w, 
             alpha_p=config.alpha_p,
             alpha_c=config.alpha_c,
             init_good=config.init_good,
             init_cap=config.init_cap,
             k_labor=config.k_labor,
             k_capital=config.k_capital,
             )
    B0 = bank(rn=config.rn, 
             pi_t=config.pi_t, 
             un=config.un, 
             alpha_pi=config.alpha_pi, 
             alpha_u=config.alpha_u, 
             num_agents=config.num_agents, 
             rate_max=config.r_max, 
             rate_min=config.r_min,
             init_assets=config.init_assets,
             )
    agents0 = init_agents(config)
    
    F0.P = np.mean([a.w*a.pc for a in agents0]) # t=0 initial price
    
    
    logs, others = [], []
    for i in range(5):
        config.seed = i
        log = {
            0:{
                'year': 0,
                'work_state': [a.l for a in agents0], 
                'pw': [a.pw for a in agents0],
                'pc': [a.pc for a in agents0],
                'wage': [a.w for a in agents0],
                'price': F0.P, 
                'rate': B0.rate, 
                'inventory': F0.G,
                'production': 0.,
                'imbalance': 0.,
                'inflation_rate': 0.,
                'taxes': 0,
                'unemployment_rate': 1-sum([a.l for a in agents0])/config.num_agents,
                'deposit': {i: 0.0 for i in range(config.num_agents)},
                'avg_wage': sum([a.w for a in agents0])/config.num_agents,
                'GDP': 0.,
                'capital': F0.capital,
                'assets': B0.assets,
                'gini': gini_coefficient([a.w for a in agents0]),
                
                
                }
            }
        F, B, agents, log = step_simulation(config, event=1, intervention=False, step=0, length=100, firm=deepcopy(F0), bank=deepcopy(B0), agents=deepcopy(agents0), log=deepcopy(log))
        F, B, agents, log = step_simulation(config, event=1, intervention=False, step=100, length=100, firm=deepcopy(F), bank=deepcopy(B), agents=deepcopy(agents), log=deepcopy(log))
        F, B, agents, log = step_simulation(config, event=1, intervention=False, step=200, length=100, firm=deepcopy(F), bank=deepcopy(B), agents=deepcopy(agents), log=deepcopy(log))
        F, B, agents, log = step_simulation(config, event=1, intervention=False, step=300, length=100, firm=deepcopy(F), bank=deepcopy(B), agents=deepcopy(agents), log=deepcopy(log))
        F, B, agents, log = step_simulation(config, event=1, intervention=False, step=400, length=100, firm=deepcopy(F), bank=deepcopy(B), agents=deepcopy(agents), log=deepcopy(log))
        F, B, agents, log = step_simulation(config, event=1, intervention=False, step=500, length=100, firm=deepcopy(F), bank=deepcopy(B), agents=deepcopy(agents), log=deepcopy(log))
        print(i, max(list(log.keys())))
        logs.append(log)
        others.append([F, B, agents])
    
    # logss = []
    # for i in range(5):
    #     config.seed = i
    #     F, B, agents, log = step_simulation(config, event=False, intervention=False, step=max(list(logs[0].keys())), length=100, firm=deepcopy(others[i][0]), bank=deepcopy(others[i][1]), agents=deepcopy(others[i][2]), log=deepcopy(logs[i]))
    #     logss.append(log)
    plot_log('./figs/log_step.png', log, config)
    # import datetime
    # plot_bar(f'./figs/bar_step_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png', logss, None, config)
    