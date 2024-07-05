import numpy as np


def beta_dist(size=1):
    '''
    truncated beta distribution for generating hourly wage
    '''
    alpha, beta = 1.5, 2
    s = np.random.beta(alpha, beta, size) * 500
    if s <= 500/168:
        return 500/168
    return s[0]

def pay_wage(agent_list:list):
    wages = []
    for a in agent_list:
        if a.l:
            # a.z = a.w * 168   # monthly income
            wages.append(a.w * 168)
        else:
            wages.append(0.)
    return wages

def taxation(wages:list[float]):
    brackets = list(np.array([0, 9700, 39475, 84200, 160725, 204100, 510300])/12) # monthly income brackets
    rates = [0.1, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37] # tax rates
    
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
    return taxes

def total_deposit(deposits:dict):
    return sum([deposits[id] for id in deposits.keys()])

def inflation(log:dict[dict]):
    '''
    通胀率 = 本年均价 - 上年均价 / 上年均价
    '''
    price_history = [log[key]['price'] for key in log.keys()]
    assert len(price_history) >= 12
    if len(price_history) < 12*2:
        return (np.mean(price_history[-12:]) - np.mean(price_history[0:-12])) / np.mean(price_history[0:-12])
    else:
        return (np.mean(price_history[-12:]) - np.mean(price_history[-12*2:-12])) / np.mean(price_history[-12*2:-12])

def GDP(log:dict):
    '''
    名义GDP = 总产量 * 价格
    计算最近一年的名义GDP
    '''
    assert len(log) >= 12
    return sum([log[key]['production'] * log[key]['price'] for key in list(log.keys())[-12:]])

if __name__ == '__main__':
    # taxes = taxation([4500, 21000, 57000, 115000, 180000, 300000, 700000])
    # print(taxes)
    pass