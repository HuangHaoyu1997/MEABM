from time import time
class Configuration:
    #####################################
    ##           干 预 参 数           ##
    #####################################
    event_start = 200
    event_end = 350
    intervent_start = 200
    intervent_end = 400
    
    #####################################
    ##           仿 真 参 数           ##
    #####################################
    seed = 124 # int(time()) # 123
    num_agents = 100
    num_time_steps = 12 * 100
    
    
    #####################################
    ##           企业模型参数           ##
    #####################################
    
    A = 1.0         # 生产率
    alpha_w = 0.05
    alpha_p = 0.10
    alpha_c = 0.05
    init_good = 50000
    init_cap = 1e7
    
    #####################################
    ##           银行模型参数           ##
    #####################################
    rn = 0.01 # 0.01         # natural interest rate 自然利率
    pi_t = 0.05 # 0.02        # target inflation rate 目标通膨率
    un = 0.05 # 0.04          # natural unemployment rate 自然失业率
    alpha_pi = 1.0
    alpha_u = 1.0
    r_min = -0.05            # minimal inter
    init_assets = 1e6
    
    #####################################
    ##          Agent模型参数          ##
    #####################################
    gamma = 0.5
    beta = 0.5
    wage_mean = 80
    wage_std = 20
    
    pw_low = 0.60
    pw_high = 1.0
    pc_low = 0.1
    pc_high = 0.4
    
    pw_delta = 0.03
    pc_delta = 0.01
    
    