import numpy as np
from config import Configuration
from src.agent import agent

def consumption(config:Configuration, agents:list[agent], good_quantity:float, price:float, deposits:dict, rng:np.random.Generator=None):
    '''
    随机顺序的消费
    return: 
    - total_consume_money: 总消费金额
    - total_consume_quantity: 总消费数量
    - deposits: 更新后的储蓄账户
    '''
    random_list = np.arange(config.num_agents)
    rng.shuffle(random_list)
    total_consume_money, total_consume_quantity = 0., 0.
    
    for idx in random_list: 
        a = agents[idx]
        intend_demand = a.pc * deposits[a.id] / price # intended demand
        demand_actual = min(intend_demand, good_quantity) # actual demand quanity
        money_actual = demand_actual * price # actual consumption
        # print('here5', deposits[a.id], intend_demand, good_quantity, money_actual)
        total_consume_quantity += demand_actual
        deposits[a.id] -= money_actual # update deposit
        good_quantity -= demand_actual # update inventory of essential goods
        total_consume_money += money_actual

    return total_consume_money, total_consume_quantity, deposits

def taxation(wages:list[float]):
    '''
    阶梯税率
    '''
    brackets = [0, 9700/120, 39475/120, 84200/120, 160725/120, 204100/120, 510300/120] # monthly income brackets
    rates =    [  0.1,   0.12,    0.22,      0.24,      0.32,      0.35,       0.37] # tax rates
    
    taxes = []
    for w in wages:
        if w <= 0: 
            taxes.append(0.)
            continue
        
        tax = 0.0
        for i in range(len(brackets) - 1):
            if w > brackets[i + 1]:
                tax += (brackets[i + 1] - brackets[i]) * rates[i]
            else:
                tax += (w - brackets[i]) * rates[i]
                break
        if w > brackets[-1]:
            tax += (w - brackets[-1]) * rates[-1]
        taxes.append(tax)
    
    wages_after_tax = [w - t for w, t in zip(wages, taxes)]
    return taxes, wages_after_tax
