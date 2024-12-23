from config import Configuration, EconomicCrisisConfig, ReconstructionConfig
from src.agent import agent
from src.firm import firm
from src.bank import bank   
from src.episode_simulation import simulation
from copy import deepcopy



if __name__ == '__main__':
    from src.utils import plot_log, plot_bar
    from config import Configuration
    from time import time
    import pickle
    name = input('Please input the name of the experiment: ')
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
    #     log = simulation(config, event=0, intervention=True)
    #     logs.append(log)
    # plot_bar('./figs/bar-no-event-no-intervention.png', logs, logs_compare=None, config=config)
    
    
    ####################################### 对 照 试 验 #######################################
    config_regulation = deepcopy(EconomicCrisisConfig()); config_regulation.seed = 123456
    config_no_regulation = deepcopy(EconomicCrisisConfig()); config_no_regulation.seed = 123456
    logs_PSRN, logs_Taylor, logs_Fixed = [], [], []
    for i in range(5):
        print(f'Simulation {i+1}/5')
        config_regulation.seed += i; config_no_regulation.seed += i
        logs_PSRN.append(simulation(config_regulation, intervention=True, rate_change=True))
        logs_Taylor.append(simulation(config_no_regulation, intervention=False, rate_change=True))
        logs_Fixed.append(simulation(config_no_regulation, intervention=False, rate_change=False))
    # plot_bar(f'./figs/{name}.pdf', logs_intervention, logs_no_intervention, config_no_regulation)
    
    with open(f'./data/logs_PSRN_{name}.pkl', 'wb') as f:
        pickle.dump(logs_PSRN, f)
    with open(f'./data/logs_Taylor_{name}.pkl', 'wb') as f:
        pickle.dump(logs_Taylor, f)
    with open(f'./data/logs_Fixed_{name}.pkl', 'wb') as f:
        pickle.dump(logs_Fixed, f)
    
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
    #     log = simulation(config, event=0, intervention=True)
    #     logs.append(log)
    # plot_bar('./figs/bar-no-event-intervention.png', logs, logs_no_event, config)
    # 
    # 