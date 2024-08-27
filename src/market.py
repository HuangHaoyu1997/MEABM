import numpy as np
from config import Configuration
from src.agent import agent

def consumption(config:Configuration, agents:list[agent], good_quantity:float, price:float, deposits:dict):
    '''
    随机顺序的消费
    return: 
    - total_consume_money: 总消费金额
    - total_consume_quantity: 总消费数量
    - deposits: 更新后的储蓄账户
    '''
    random_list = np.arange(config.num_agents)
    np.random.shuffle(random_list)
    total_consume_money, total_consume_quantity = 0., 0.
    
    for idx in random_list: 
        a = agents[idx]
        intend_demand = a.pc * deposits[a.id] / price # intended demand
        demand_actual = min(intend_demand, good_quantity) # actual demand quanity
        money_actual = demand_actual * price # actual consumption
        total_consume_quantity += demand_actual
        deposits[a.id] -= money_actual # update deposit
        good_quantity -= demand_actual # update inventory of essential goods
        total_consume_money += money_actual

    return total_consume_money, total_consume_quantity, deposits