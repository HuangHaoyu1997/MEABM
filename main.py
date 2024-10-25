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
    config_no_regulation = deepcopy(EconomicCrisisConfig())
    config_no_regulation.seed = 123456
    config_regulation = deepcopy(EconomicCrisisConfig()); config_regulation.seed = 123456
    # config_no_regulation = EconomicCrisisConfig(); config_no_regulation.seed = 123456
    logs_intervention, logs_no_intervention = [], []
    for i in range(5):
        print(f'Simulation {i+1}/5')
        config_regulation.seed += i; config_no_regulation.seed += i
        logs_intervention.append(simulation(config_regulation, intervention=True))
        logs_no_intervention.append(simulation(config_no_regulation, intervention=False))
    plot_bar('./figs/bar-event-intervention1.svg', logs_intervention, logs_no_intervention, config_no_regulation)
    
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