import numpy as np
from config import Configuration
from src.agent import agent, gauss_dist

def gini_coefficient(income_list:list[float]) -> float:
    sorted_income = sorted(income_list) # 排序收入列表
    n = len(sorted_income)
    
    total_income = sum(sorted_income) # 总收入
    cumulative_income = [sum(sorted_income[:i + 1]) for i in range(n)] # 累计收入

    # 计算基尼系数
    gini = 1 - 2 * sum(cumulative_income) / (n * total_income)
    return gini


def split_img(img_path:str):
    import cv2
    import matplotlib.pyplot as plt
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    price_fig =         img[30:1550, 50:1850, :]
    interest_rate_fig = img[30:1550, 1850:3650, :]
    employment_fig =    img[30:1550, 3650:5400, :]
    inflation_fig =     img[30:1550, 5400:7200, :]
    gini_fig =          img[30:1550, 7200:, :]
    
    imbalance_fig =  img[1600:3150, 0:1800, :]
    capital_fig =    img[1600:3150, 1850:3600, :]
    production_fig = img[1600:3150, 3600:5400, :]
    gdp_fig =        img[1600:3150, 5450:7200, :]
    
    avg_deposit_fig = img[3200:-50, 50:1850, :]
    avg_wage_fig =    img[3200:-50, 1850:3600, :]
    avg_tax_fig =     img[3200:-50, 3600:5400, :]
    std_wage_fig =    img[3200:-50, 5450:7200, :]
    # plt.imshow(std_wage_fig)
    # plt.axis('off')
    # plt.show()
    return [price_fig, interest_rate_fig, employment_fig, inflation_fig, gini_fig, imbalance_fig, capital_fig, production_fig, gdp_fig, avg_deposit_fig, avg_wage_fig, avg_tax_fig, std_wage_fig]

def init_agents(config:Configuration) -> list[agent]:
    agent_list = [agent(id=i, 
                  pw=np.random.uniform(config.pw_low, config.pw_high), 
                  pc=np.random.uniform(config.pc_low, config.pc_high), 
                  gamma=config.gamma, 
                  beta=config.gamma,
                  pw_delta=config.pw_delta,
                  pc_delta=config.pc_delta,) for i in range(config.num_agents)]
    wages = gauss_dist(config.wage_mean, config.wage_std, config.num_agents)
    for a, w in zip(agent_list, wages):
        a.w = w
    return agent_list

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
    
    wages_after_tax = [w - t for w, t in zip(wages, taxes)]
    return taxes, wages_after_tax

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

def total_intended_demand(agent_list:list[agent], P:float, deposits:dict) -> float:
    '''
    社会总预期需求 = 消费比例 * 储蓄量 / 价格
    '''
    cnt = np.sum([a.pc * deposits[a.id] / P for a in agent_list])
    return cnt

def imbalance(agent_list:list[agent], P:float, G:float, deposits:dict) -> float:
    '''
    计算 预期需求与实际产量之间的不均衡
    >0, 需求 > 产量
    <0, 需求 < 产量
    '''
    D = total_intended_demand(agent_list, P, deposits)
    # print('imbalance:', D, firm.G)
    phi_bar = (D - G) / max(D, G)
    return phi_bar

def plot_bar(img_name:str, logs:list[dict], logs_compare:list[dict], config:Configuration):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 5, figsize=(30, 16))
    # fig.suptitle('xxx')
    for i in range(3):
        for j in range(5):
            axs[i, j].set_xlabel('Time / Month'); axs[i, j].grid()
            axs[i, j].axvline(x=config.event_start, color='r', linestyle='--')
            axs[i, j].axvline(x=config.event_end, color='r', linestyle='--')
            axs[i, j].set_xlim(-20, config.num_time_steps+20)
    
    prices = np.array([[log[key]['price'] for key in log.keys()] for log in logs])
    prices_mean = np.mean(prices, axis=0)
    x = list(range(len(prices[0])))
    prices_min = prices_mean - np.std(prices, axis=0)
    prices_max = prices_mean + np.std(prices, axis=0)
    
    
    rates = np.array([[log[key]['rate'] for key in log.keys()] for log in logs])
    rates_mean = np.mean(rates, axis=0)
    rates_min = rates_mean - np.std(rates, axis=0)
    rates_max = rates_mean + np.std(rates, axis=0)
    
    um_rates = [[1-log[key]['unemployment_rate'] for key in log.keys()] for log in logs]
    um_rates_mean = np.mean(um_rates, axis=0)
    um_rates_min = um_rates_mean - np.std(um_rates, axis=0)
    um_rates_max = um_rates_mean + np.std(um_rates, axis=0)
    
    inflation_rates = [[log[key]['inflation_rate'] for key in log.keys()] for log in logs]
    inflation_rates_mean = np.mean(inflation_rates, axis=0)
    inflation_rates_min = inflation_rates_mean - np.std(inflation_rates, axis=0)
    inflation_rates_max = inflation_rates_mean + np.std(inflation_rates, axis=0)
    
    ginis = [[log[key]['gini'] for key in log.keys()] for log in logs]
    ginis_mean = np.mean(ginis, axis=0)
    ginis_min = ginis_mean - np.std(ginis, axis=0)
    ginis_max = ginis_mean + np.std(ginis, axis=0)
    
    
    imbas = [[log[key]['imbalance'] for key in log.keys()] for log in logs]
    imbas_mean = np.mean(imbas, axis=0)
    imbas_min = imbas_mean - np.std(imbas, axis=0)
    imbas_max = imbas_mean + np.std(imbas, axis=0)
    
    capitals = [[log[key]['capital'] for key in log.keys()] for log in logs]
    capitals_mean = np.mean(capitals, axis=0)
    capitals_min = capitals_mean - np.std(capitals, axis=0)
    capitals_max = capitals_mean + np.std(capitals, axis=0)
    
    assets = [[log[key]['assets'] for key in log.keys()] for log in logs]
    assets_mean = np.mean(assets, axis=0)
    assets_min = assets_mean - np.std(assets, axis=0)
    assets_max = assets_mean + np.std(assets, axis=0)
    
    productions = [[log[key]['production'] for key in log.keys()] for log in logs]
    productions_mean = np.mean(productions, axis=0)
    productions_min = productions_mean - np.std(productions, axis=0)
    productions_max = productions_mean + np.std(productions, axis=0)
    
    GDPs = [[log[key]['GDP'] for key in log.keys()] for log in logs]
    GDPs_mean = np.mean(GDPs, axis=0)
    GDPs_min = GDPs_mean - np.std(GDPs, axis=0)
    GDPs_max = GDPs_mean + np.std(GDPs, axis=0)
    
    deposits = [[total_deposit(log[key]['deposit'])/config.num_agents for key in log.keys()] for log in logs]
    deposits_mean = np.mean(deposits, axis=0)
    deposits_min = deposits_mean - np.std(deposits, axis=0)
    deposits_max = deposits_mean + np.std(deposits, axis=0)
    
    avg_wages = [[log[key]['avg_wage'] for key in log.keys()] for log in logs]
    avg_wages_mean = np.mean(avg_wages, axis=0)
    avg_wages_min = avg_wages_mean - np.std(avg_wages, axis=0)
    avg_wages_max = avg_wages_mean + np.std(avg_wages, axis=0)
    
    taxes = [[log[key]['taxes']/config.num_agents for key in log.keys()] for log in logs]
    taxes_mean = np.mean(taxes, axis=0)
    taxes_min = taxes_mean - np.std(taxes, axis=0)
    taxes_max = taxes_mean + np.std(taxes, axis=0)
    
    wage_stds = [[np.std(log[key]['wage']) for key in log.keys()] for log in logs]
    wage_stds_mean = np.mean(wage_stds, axis=0)
    wage_stds_min = wage_stds_mean - np.std(wage_stds, axis=0)
    wage_stds_max = wage_stds_mean + np.std(wage_stds, axis=0)
    
    axs[0, 0].plot(x, prices_mean); axs[0, 0].set_ylabel('Price', fontsize=14)
    axs[0, 0].fill_between(x, prices_min, prices_max, color='red', alpha=0.3)
    
    axs[0, 1].plot(x, rates_mean); axs[0, 1].set_ylabel('Interest rate', fontsize=14)
    axs[0, 1].fill_between(x, rates_min, rates_max, color='red', alpha=0.3)
    
    axs[0, 2].plot(x, um_rates_mean); axs[0, 2].set_ylabel('Employment rate', fontsize=14)
    axs[0, 2].fill_between(x, um_rates_min, um_rates_max, color='red', alpha=0.3)
    
    axs[0, 3].plot(x, inflation_rates_mean); axs[0, 3].set_ylabel('Inflation rate', fontsize=14)
    axs[0, 3].fill_between(x, inflation_rates_min, inflation_rates_max, color='red', alpha=0.3)
    
    axs[0, 4].plot(x, ginis_mean); axs[0, 4].set_ylabel('Gini coefficient', fontsize=14)
    axs[0, 4].fill_between(x, ginis_min, ginis_max, color='red', alpha=0.3)
    
    axs[1, 0].plot(x, imbas_mean); axs[1, 0].set_ylabel('Imbalance: Demand - Supply', fontsize=14)
    axs[1, 0].fill_between(x, imbas_min, imbas_max, color='red', alpha=0.3)
    
    axs[1, 1].plot(x, capitals_mean, label='cap'); axs[1, 1].set_ylabel('Capital & Assets', fontsize=14)
    axs[1, 1].fill_between(x, capitals_min, capitals_max, color='red', alpha=0.3)
    axs[1, 1].plot(x, assets_mean, label='ass')
    axs[1, 1].fill_between(x, assets_min, assets_max, color='blue', alpha=0.3); axs[1, 1].legend()
    
    axs[1, 2].plot(x, productions_mean); axs[1, 2].set_ylabel('Production', fontsize=14)
    axs[1, 2].fill_between(x, productions_min, productions_max, color='red', alpha=0.3)
    
    axs[1, 3].plot(x, GDPs_mean); axs[1, 3].set_ylabel('Nominal GDP', fontsize=14)
    axs[1, 3].fill_between(x, GDPs_min, GDPs_max, color='red', alpha=0.3)
    
    axs[2, 0].plot(x, deposits_mean); axs[2, 0].set_ylabel('Deposit per capita', fontsize=14)
    axs[2, 0].fill_between(x, deposits_min, deposits_max, color='red', alpha=0.3)
    
    axs[2, 1].plot(x, avg_wages_mean); axs[2, 1].set_ylabel('Avg wage', fontsize=14)
    axs[2, 1].fill_between(x, avg_wages_min, avg_wages_max, color='red', alpha=0.3)
    
    axs[2, 2].plot(x, taxes_mean); axs[2, 2].set_ylabel('Avg tax revenue per capita', fontsize=14)
    axs[2, 2].fill_between(x, taxes_min, taxes_max, color='red', alpha=0.3)
    
    axs[2, 3].plot(x, wage_stds_mean); axs[2, 3].set_ylabel('Wage std', fontsize=14)
    axs[2, 3].fill_between(x, wage_stds_min, wage_stds_max, color='red', alpha=0.3)
    if logs_compare is not None:
        prices = np.array([[log[key]['price'] for key in log.keys()] for log in logs_compare])
        prices_mean = np.mean(prices, axis=0)
        x = list(range(len(prices[0])))
        prices_min = prices_mean - np.std(prices, axis=0)
        prices_max = prices_mean + np.std(prices, axis=0)
        
        
        rates = np.array([[log[key]['rate'] for key in log.keys()] for log in logs_compare])
        rates_mean = np.mean(rates, axis=0)
        rates_min = rates_mean - np.std(rates, axis=0)
        rates_max = rates_mean + np.std(rates, axis=0)
        
        um_rates = [[1-log[key]['unemployment_rate'] for key in log.keys()] for log in logs_compare]
        um_rates_mean = np.mean(um_rates, axis=0)
        um_rates_min = um_rates_mean - np.std(um_rates, axis=0)
        um_rates_max = um_rates_mean + np.std(um_rates, axis=0)
        
        inflation_rates = [[log[key]['inflation_rate'] for key in log.keys()] for log in logs_compare]
        inflation_rates_mean = np.mean(inflation_rates, axis=0)
        inflation_rates_min = inflation_rates_mean - np.std(inflation_rates, axis=0)
        inflation_rates_max = inflation_rates_mean + np.std(inflation_rates, axis=0)
        
        ginis = [[log[key]['gini'] for key in log.keys()] for log in logs_compare]
        ginis_mean = np.mean(ginis, axis=0)
        ginis_min = ginis_mean - np.std(ginis, axis=0)
        ginis_max = ginis_mean + np.std(ginis, axis=0)
    
        imbas = [[log[key]['imbalance'] for key in log.keys()] for log in logs_compare]
        imbas_mean = np.mean(imbas, axis=0)
        imbas_min = imbas_mean - np.std(imbas, axis=0)
        imbas_max = imbas_mean + np.std(imbas, axis=0)
        
        capitals = [[log[key]['capital']+log[key]['assets'] for key in log.keys()] for log in logs_compare]
        capitals_mean = np.mean(capitals, axis=0)
        capitals_min = capitals_mean - np.std(capitals, axis=0)
        capitals_max = capitals_mean + np.std(capitals, axis=0)
        
        assets = [[log[key]['assets'] for key in log.keys()] for log in logs_compare]
        assets_mean = np.mean(assets, axis=0)
        assets_min = assets_mean - np.std(assets, axis=0)
        assets_max = assets_mean + np.std(assets, axis=0)
    
        productions = [[log[key]['production'] for key in log.keys()] for log in logs_compare]
        productions_mean = np.mean(productions, axis=0)
        productions_min = productions_mean - np.std(productions, axis=0)
        productions_max = productions_mean + np.std(productions, axis=0)
        
        GDPs = [[log[key]['GDP'] for key in log.keys()] for log in logs_compare]
        GDPs_mean = np.mean(GDPs, axis=0)
        GDPs_min = GDPs_mean - np.std(GDPs, axis=0)
        GDPs_max = GDPs_mean + np.std(GDPs, axis=0)
        
        deposits = [[total_deposit(log[key]['deposit'])/config.num_agents for key in log.keys()] for log in logs_compare]
        deposits_mean = np.mean(deposits, axis=0)
        deposits_min = deposits_mean - np.std(deposits, axis=0)
        deposits_max = deposits_mean + np.std(deposits, axis=0)
        
        avg_wages = [[log[key]['avg_wage'] for key in log.keys()] for log in logs_compare]
        avg_wages_mean = np.mean(avg_wages, axis=0)
        avg_wages_min = avg_wages_mean - np.std(avg_wages, axis=0)
        avg_wages_max = avg_wages_mean + np.std(avg_wages, axis=0)
        
        taxes = [[log[key]['taxes']/config.num_agents for key in log.keys()] for log in logs_compare]
        taxes_mean = np.mean(taxes, axis=0)
        taxes_min = taxes_mean - np.std(taxes, axis=0)
        taxes_max = taxes_mean + np.std(taxes, axis=0)
        
        wage_stds = [[np.std(log[key]['wage']) for key in log.keys()] for log in logs_compare]
        wage_stds_mean = np.mean(wage_stds, axis=0)
        wage_stds_min = wage_stds_mean - np.std(wage_stds, axis=0)
        wage_stds_max = wage_stds_mean + np.std(wage_stds, axis=0)
        
        axs[0, 0].plot(x, prices_mean); axs[0, 0].set_ylabel('Price', fontsize=14)
        axs[0, 0].fill_between(x, prices_min, prices_max, color='gray', alpha=0.3)
        
        axs[0, 1].plot(x, rates_mean); axs[0, 1].set_ylabel('Interest rate', fontsize=14)
        axs[0, 1].fill_between(x, rates_min, rates_max, color='gray', alpha=0.3)
        
        axs[0, 2].plot(x, um_rates_mean); axs[0, 2].set_ylabel('Employment rate', fontsize=14)
        axs[0, 2].fill_between(x, um_rates_min, um_rates_max, color='gray', alpha=0.3)
        
        axs[0, 3].plot(x, inflation_rates_mean); axs[0, 3].set_ylabel('Inflation rate', fontsize=14)
        axs[0, 3].fill_between(x, inflation_rates_min, inflation_rates_max, color='gray', alpha=0.3)
        
        axs[0, 4].plot(x, ginis_mean); axs[0, 4].set_ylabel('Gini coefficient', fontsize=14)
        axs[0, 4].fill_between(x, ginis_min, ginis_max, color='gray', alpha=0.3)
    
        axs[1, 0].plot(x, imbas_mean); axs[1, 0].set_ylabel('Imbalance: Demand - Supply', fontsize=14)
        axs[1, 0].fill_between(x, imbas_min, imbas_max, color='gray', alpha=0.3)
        
        axs[1, 1].plot(x, capitals_mean, label='cap'); axs[1, 1].set_ylabel('Capital & Assets', fontsize=14)
        axs[1, 1].fill_between(x, capitals_min, capitals_max, color='gray', alpha=0.3)
        axs[1, 1].plot(x, assets_mean, label='ass')
        axs[1, 1].fill_between(x, assets_min, assets_max, color='gray', alpha=0.3); axs[1, 1].legend()
    
        axs[1, 2].plot(x, productions_mean); axs[1, 2].set_ylabel('Production', fontsize=14)
        axs[1, 2].fill_between(x, productions_min, productions_max, color='gray', alpha=0.3)
        
        axs[1, 3].plot(x, GDPs_mean); axs[1, 3].set_ylabel('Nominal GDP', fontsize=14)
        axs[1, 3].fill_between(x, GDPs_min, GDPs_max, color='gray', alpha=0.3)
        
        axs[2, 0].plot(x, deposits_mean); axs[2, 0].set_ylabel('Deposit per capita', fontsize=14)
        axs[2, 0].fill_between(x, deposits_min, deposits_max, color='gray', alpha=0.3)
        
        axs[2, 1].plot(x, avg_wages_mean); axs[2, 1].set_ylabel('Avg wage', fontsize=14)
        axs[2, 1].fill_between(x, avg_wages_min, avg_wages_max, color='gray', alpha=0.3)
        
        axs[2, 2].plot(x, taxes_mean); axs[2, 2].set_ylabel('Avg tax revenue per capita', fontsize=14)
        axs[2, 2].fill_between(x, taxes_min, taxes_max, color='gray', alpha=0.3)
        
        axs[2, 3].plot(x, wage_stds_mean); axs[2, 3].set_ylabel('Wage std', fontsize=14)
        axs[2, 3].fill_between(x, wage_stds_min, wage_stds_max, color='gray', alpha=0.3)
    plt.tight_layout()
    plt.savefig(img_name, dpi=300)

def plot_log(img_name:str, log:dict, config:Configuration):
    import matplotlib.pyplot as plt
    
    fig, axs = plt.subplots(3, 5, figsize=(30, 16))
    # fig.suptitle('xxx')
    for i in range(3):
        for j in range(5):
            axs[i, j].set_xlabel('Time / Month'); axs[i, j].grid()
            axs[i, j].axvline(x=config.event_start, color='r', linestyle='--')
            axs[i, j].axvline(x=config.event_end, color='r', linestyle='--')
            axs[i, j].set_xlim(-20, config.num_time_steps+20)
    
    price_history = [log[key]['price'] for key in log.keys()]
    rate_history = [log[key]['rate'] for key in log.keys()]
    um_rate_history = [1-log[key]['unemployment_rate'] for key in log.keys()]
    inflation_history = [log[key]['inflation_rate'] for key in log.keys()]
    gini_history = [log[key]['gini'] for key in log.keys()]
    imba_history = [log[key]['imbalance'] for key in log.keys()]
    capital_history = [log[key]['capital'] for key in log.keys()]
    production_history = [log[key]['production'] for key in log.keys()]
    GDP_history = [log[key]['GDP'] for key in log.keys()]
    deposit_history = [total_deposit(log[key]['deposit'])/config.num_agents for key in log.keys()]
    wage_history = [log[key]['avg_wage'] for key in log.keys()]
    taxes_history = [log[key]['taxes']/config.num_agents for key in log.keys()]
    wage_std_history = [np.std(log[key]['wage']) for key in log.keys()]
    axs[0, 0].plot(price_history);      axs[0, 0].set_ylabel('Price', fontsize=14)
    axs[0, 1].plot(rate_history);       axs[0, 1].set_ylabel('Interest rate', fontsize=14)
    axs[0, 2].plot(um_rate_history);    axs[0, 2].set_ylabel('Employment rate', fontsize=14)
    axs[0, 3].plot(inflation_history);  axs[0, 3].set_ylabel('Inflation rate', fontsize=14)
    axs[0, 4].plot(gini_history);       axs[0, 4].set_ylabel('Gini coefficient', fontsize=14)
    axs[1, 0].plot(imba_history);       axs[1, 0].set_ylabel('Imbalance: Demand - Supply', fontsize=14)
    axs[1, 1].plot(capital_history);    axs[1, 1].set_ylabel('Capital', fontsize=14)
    axs[1, 2].plot(production_history); axs[1, 2].set_ylabel('Production', fontsize=14)
    axs[1, 3].plot(GDP_history);        axs[1, 3].set_ylabel('Nominal GDP', fontsize=14)
    axs[2, 0].plot(deposit_history);    axs[2, 0].set_ylabel('Deposit per capita', fontsize=14)
    axs[2, 1].plot(wage_history);       axs[2, 1].set_ylabel('Avg wage', fontsize=14)
    axs[2, 2].plot(taxes_history);      axs[2, 2].set_ylabel('Avg tax revenue per capita', fontsize=14)
    axs[2, 3].plot(wage_std_history);   axs[2, 3].set_ylabel('Wage std', fontsize=14)
    plt.tight_layout()
    plt.savefig(img_name, dpi=300)
    # plt.show()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # taxes = taxation([4500, 21000, 57000, 115000, 180000, 300000, 700000])
    # print(taxes)
    
    config = Configuration()
    # agents = init_agents(config)
    # plt.hist([a.w for a in agents], bins=20)
    # plt.show()
    
    split_img(img_path='./figs/bar-event-intervention.png')
    
    # 示例：创建100个智能体的随机存款
    import random

    # 生成随机存款
    # random_income = [random.gauss(10, 0.23) for _ in range(100)]

    # wages = gauss_dist(config.wage_mean, config.wage_std, config.num_agents)
    # gini = gini_coefficient(wages)
    # print("基尼系数:", gini)