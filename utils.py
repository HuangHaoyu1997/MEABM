import numpy as np
from config import Configuration
from src.agent import agent


def init_agents(config:Configuration) -> list[agent]:
    return [agent(id=i, 
                  pw=np.random.uniform(config.pw_low, config.pw_high), 
                  pc=np.random.uniform(config.pc_low, config.pc_high), 
                  gamma=config.gamma, 
                  beta=config.gamma) for i in range(config.num_agents)]

def taxation(wages:list[float]):
    '''
    阶梯税率
    '''
    brackets = [0, 9700/12, 39475/12, 84200/12, 160725/12, 204100/12, 510300/12] # monthly income brackets
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
        return (np.mean(price_history[-12:]) - price_history[-12]) / price_history[-12]
    else:
        return (np.mean(price_history[-12:]) - np.mean(price_history[-12*2:-12])) / np.mean(price_history[-12*2:-12])

def GDP(log:dict):
    '''
    名义GDP = 总产量 * 价格
    计算最近一年的名义GDP
    '''
    assert len(log) >= 12
    return sum([log[key]['production'] * log[key]['price'] for key in list(log.keys())[-12:]])

def unemployment(log:dict):
    '''
    统计最近一年的失业率
    '''
    work_state_history = [log[key]['work_state'] for key in log.keys()]
    unemployment_cnt = 0
    tmp_states = work_state_history[-12:]
    for state in tmp_states:
        unemployment_cnt += state.count(0)
    return unemployment_cnt / (12 * len(state))

def plot_bar(img_name:str, logs:list[dict], logs_compare:list[dict], config:Configuration):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 4, figsize=(24, 16))
    fig.suptitle('xxx')
    for i in range(3):
        for j in range(4):
            axs[i, j].set_xlabel('Time / Month'); axs[i, j].grid()
            axs[i, j].axvline(x=500, color='r', linestyle='--')
            axs[i, j].axvline(x=900, color='r', linestyle='--')
            
    
    prices = np.array([[log[key]['price'] for key in log.keys()] for log in logs])
    prices_mean = np.mean(prices, axis=0)
    prices_std = np.std(prices, axis=0)
    x = list(range(len(prices[0])))
    prices_min = prices_mean - prices_std
    prices_max = prices_mean + prices_std
    
    
    rates = np.array([[log[key]['rate'] for key in log.keys()] for log in logs])
    rates_mean = np.mean(rates, axis=0)
    rates_std = np.std(rates, axis=0)
    rates_min = rates_mean - rates_std
    rates_max = rates_mean + rates_std
    
    um_rates = [[1-log[key]['unemployment_rate'] for key in log.keys()] for log in logs]
    um_rates_mean = np.mean(um_rates, axis=0)
    um_rates_std = np.std(um_rates, axis=0)
    um_rates_min = um_rates_mean - um_rates_std
    um_rates_max = um_rates_mean + um_rates_std
    
    inflation_rates = [[log[key]['inflation_rate'] for key in log.keys()] for log in logs]
    inflation_rates_mean = np.mean(inflation_rates, axis=0)
    inflation_rates_std = np.std(inflation_rates, axis=0)
    inflation_rates_min = inflation_rates_mean - inflation_rates_std
    inflation_rates_max = inflation_rates_mean + inflation_rates_std
    
    
    imbas = [[log[key]['imbalance'] for key in log.keys()] for log in logs]
    imbas_mean = np.mean(imbas, axis=0)
    imbas_std = np.std(imbas, axis=0)
    imbas_min = imbas_mean - imbas_std
    imbas_max = imbas_mean + imbas_std
    
    productions = [[log[key]['production'] for key in log.keys()] for log in logs]
    productions_mean = np.mean(productions, axis=0)
    productions_std = np.std(productions, axis=0)
    productions_min = productions_mean - productions_std
    productions_max = productions_mean + productions_std
    
    GDPs = [[log[key]['GDP'] for key in log.keys()] for log in logs]
    GDPs_mean = np.mean(GDPs, axis=0)
    GDPs_std = np.std(GDPs, axis=0)
    GDPs_min = GDPs_mean - GDPs_std
    GDPs_max = GDPs_mean + GDPs_std
    
    deposits = [[total_deposit(log[key]['deposit'])/config.num_agents for key in log.keys()] for log in logs]
    deposits_mean = np.mean(deposits, axis=0)
    deposits_std = np.std(deposits, axis=0)
    deposits_min = deposits_mean - deposits_std
    deposits_max = deposits_mean + deposits_std
    
    avg_wages = [[log[key]['avg_wage'] for key in log.keys()] for log in logs]
    avg_wages_mean = np.mean(avg_wages, axis=0)
    avg_wages_std = np.std(avg_wages, axis=0)
    avg_wages_min = avg_wages_mean - avg_wages_std
    avg_wages_max = avg_wages_mean + avg_wages_std
    
    axs[0, 0].plot(x, prices_mean); axs[0, 0].set_ylabel('Price', fontsize=14)
    axs[0, 0].fill_between(x, prices_min, prices_max, color='red', alpha=0.3)
    
    axs[0, 1].plot(x, rates_mean); axs[0, 1].set_ylabel('Interest rate', fontsize=14)
    axs[0, 1].fill_between(x, rates_min, rates_max, color='red', alpha=0.3)
    
    axs[0, 2].plot(x, um_rates_mean); axs[0, 2].set_ylabel('Employment rate', fontsize=14)
    axs[0, 2].fill_between(x, um_rates_min, um_rates_max, color='red', alpha=0.3)
    
    axs[0, 3].plot(x, inflation_rates_mean); axs[0, 3].set_ylabel('Inflation rate', fontsize=14)
    axs[0, 3].fill_between(x, inflation_rates_min, inflation_rates_max, color='red', alpha=0.3)
    
    axs[1, 0].plot(x, imbas_mean); axs[1, 0].set_ylabel('Imbalance: Demand - Supply', fontsize=14)
    axs[1, 0].fill_between(x, imbas_min, imbas_max, color='red', alpha=0.3)
    
    axs[1, 2].plot(x, productions_mean); axs[1, 2].set_ylabel('Production', fontsize=14)
    axs[1, 2].fill_between(x, productions_min, productions_max, color='red', alpha=0.3)
    
    axs[1, 3].plot(x, GDPs_mean); axs[1, 3].set_ylabel('Nominal GDP', fontsize=14)
    axs[1, 3].fill_between(x, GDPs_min, GDPs_max, color='red', alpha=0.3)
    
    axs[2, 0].plot(x, deposits_mean); axs[2, 0].set_ylabel('Deposit per capita', fontsize=14)
    axs[2, 0].fill_between(x, deposits_min, deposits_max, color='red', alpha=0.3)
    
    axs[2, 1].plot(x, avg_wages_mean); axs[2, 1].set_ylabel('Avg wage', fontsize=14)
    axs[2, 1].fill_between(x, avg_wages_min, avg_wages_max, color='red', alpha=0.3)
    if logs_compare is not None:
        prices = np.array([[log[key]['price'] for key in log.keys()] for log in logs_compare])
        prices_mean = np.mean(prices, axis=0)
        prices_std = np.std(prices, axis=0)
        x = list(range(len(prices[0])))
        prices_min = prices_mean - prices_std
        prices_max = prices_mean + prices_std
        
        
        rates = np.array([[log[key]['rate'] for key in log.keys()] for log in logs_compare])
        rates_mean = np.mean(rates, axis=0)
        rates_std = np.std(rates, axis=0)
        rates_min = rates_mean - rates_std
        rates_max = rates_mean + rates_std
        
        um_rates = [[1-log[key]['unemployment_rate'] for key in log.keys()] for log in logs_compare]
        um_rates_mean = np.mean(um_rates, axis=0)
        um_rates_std = np.std(um_rates, axis=0)
        um_rates_min = um_rates_mean - um_rates_std
        um_rates_max = um_rates_mean + um_rates_std
        
        inflation_rates = [[log[key]['inflation_rate'] for key in log.keys()] for log in logs_compare]
        inflation_rates_mean = np.mean(inflation_rates, axis=0)
        inflation_rates_std = np.std(inflation_rates, axis=0)
        inflation_rates_min = inflation_rates_mean - inflation_rates_std
        inflation_rates_max = inflation_rates_mean + inflation_rates_std
        
        imbas = [[log[key]['imbalance'] for key in log.keys()] for log in logs_compare]
        imbas_mean = np.mean(imbas, axis=0)
        imbas_std = np.std(imbas, axis=0)
        imbas_min = imbas_mean - imbas_std
        imbas_max = imbas_mean + imbas_std
        
        productions = [[log[key]['production'] for key in log.keys()] for log in logs_compare]
        productions_mean = np.mean(productions, axis=0)
        productions_std = np.std(productions, axis=0)
        productions_min = productions_mean - productions_std
        productions_max = productions_mean + productions_std
        
        GDPs = [[log[key]['GDP'] for key in log.keys()] for log in logs_compare]
        GDPs_mean = np.mean(GDPs, axis=0)
        GDPs_std = np.std(GDPs, axis=0)
        GDPs_min = GDPs_mean - GDPs_std
        GDPs_max = GDPs_mean + GDPs_std
        
        deposits = [[total_deposit(log[key]['deposit'])/config.num_agents for key in log.keys()] for log in logs_compare]
        deposits_mean = np.mean(deposits, axis=0)
        deposits_std = np.std(deposits, axis=0)
        deposits_min = deposits_mean - deposits_std
        deposits_max = deposits_mean + deposits_std
        
        avg_wages = [[log[key]['avg_wage'] for key in log.keys()] for log in logs_compare]
        avg_wages_mean = np.mean(avg_wages, axis=0)
        avg_wages_std = np.std(avg_wages, axis=0)
        avg_wages_min = avg_wages_mean - avg_wages_std
        avg_wages_max = avg_wages_mean + avg_wages_std
        
        axs[0, 0].plot(x, prices_mean); axs[0, 0].set_ylabel('Price', fontsize=14)
        axs[0, 0].fill_between(x, prices_min, prices_max, color='gray', alpha=0.3)
        
        axs[0, 1].plot(x, rates_mean); axs[0, 1].set_ylabel('Interest rate', fontsize=14)
        axs[0, 1].fill_between(x, rates_min, rates_max, color='gray', alpha=0.3)
        
        axs[0, 2].plot(x, um_rates_mean); axs[0, 2].set_ylabel('Employment rate', fontsize=14)
        axs[0, 2].fill_between(x, um_rates_min, um_rates_max, color='gray', alpha=0.3)
        
        axs[0, 3].plot(x, inflation_rates_mean); axs[0, 3].set_ylabel('Inflation rate', fontsize=14)
        axs[0, 3].fill_between(x, inflation_rates_min, inflation_rates_max, color='gray', alpha=0.3)
        
        axs[1, 0].plot(x, imbas_mean); axs[1, 0].set_ylabel('Imbalance: Demand - Supply', fontsize=14)
        axs[1, 0].fill_between(x, imbas_min, imbas_max, color='gray', alpha=0.3)
        
        axs[1, 2].plot(x, productions_mean); axs[1, 2].set_ylabel('Production', fontsize=14)
        axs[1, 2].fill_between(x, productions_min, productions_max, color='gray', alpha=0.3)
        
        axs[1, 3].plot(x, GDPs_mean); axs[1, 3].set_ylabel('Nominal GDP', fontsize=14)
        axs[1, 3].fill_between(x, GDPs_min, GDPs_max, color='gray', alpha=0.3)
        
        axs[2, 0].plot(x, deposits_mean); axs[2, 0].set_ylabel('Deposit per capita', fontsize=14)
        axs[2, 0].fill_between(x, deposits_min, deposits_max, color='gray', alpha=0.3)
        
        axs[2, 1].plot(x, avg_wages_mean); axs[2, 1].set_ylabel('Avg wage', fontsize=14)
        axs[2, 1].fill_between(x, avg_wages_min, avg_wages_max, color='gray', alpha=0.3)
    plt.tight_layout()
    plt.savefig(img_name, dpi=300)

def plot_log(img_name:str, log:dict, config:Configuration):
    import matplotlib.pyplot as plt
    
    fig, axs = plt.subplots(3, 4, figsize=(24, 16))
    fig.suptitle('xxx')
    for i in range(3):
        for j in range(4):
            axs[i, j].set_xlabel('Time / Month'); axs[i, j].grid()
    
    price_history = [log[key]['price'] for key in log.keys()]
    rate_history = [log[key]['rate'] for key in log.keys()]
    imba_history = [log[key]['imbalance'] for key in log.keys()]
    taxes_history = [log[key]['taxes']/config.num_agents for key in log.keys()]
    
    axs[0, 0].plot(price_history); axs[0, 0].set_ylabel('Price')
    
    axs[0, 1].plot(rate_history); axs[0, 1].set_ylabel('Interest rate')
    
    axs[0, 2].plot([1-log[key]['unemployment_rate'] for key in log.keys()]); axs[0, 2].set_ylabel('Employment rate')
    
    axs[0, 3].plot([log[key]['GDP'] for key in log.keys()]); axs[0, 3].set_ylabel('Nominal GDP')
    
    axs[1, 0].plot(imba_history); axs[1, 0].set_ylabel('Imbalance: Demand - Supply')
    
    axs[1, 1].plot(taxes_history); axs[1, 1].set_ylabel('Avg tax revenue per capita')
    
    axs[1, 2].plot([log[key]['production'] for key in log.keys()]); axs[1, 2].set_ylabel('Production')
    
    axs[2, 0].plot([total_deposit(log[key]['deposit'])/config.num_agents for key in log.keys()]); axs[2, 0].set_ylabel('Deposit per capita')
    
    axs[2, 1].plot([log[key]['avg_wage'] for key in log.keys()]); axs[2, 1].set_ylabel('Avg wage')
    
    axs[2, 2].plot([log[key]['inflation_rate'] for key in log.keys()]); axs[2, 2].set_ylabel('Inflation rate')
    
    plt.tight_layout()
    plt.savefig(img_name, dpi=300)
    # plt.show()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # taxes = taxation([4500, 21000, 57000, 115000, 180000, 300000, 700000])
    # print(taxes)
    config = Configuration()
    agents = init_agents(config)
    plt.hist([a.w for a in agents], bins=20)
    plt.show()