from time import time
class Configuration:
    #####################################
    ##           仿 真 参 数           ##
    #####################################
    seed = int(time()) # 123
    num_agents = 200
    num_time_steps = 12 * 100
    
    
    #####################################
    ##           企业模型参数           ##
    #####################################
    
    A = 1.0         # 生产率
    alpha_w = 0.05
    alpha_p = 0.10
    
    #####################################
    ##           银行模型参数           ##
    #####################################
    rn = 0.05 # 0.01         # natural interest rate 自然利率
    pi_t = 0.1 # 0.02        # target inflation rate 目标通膨率
    un = 0.2 # 0.04          # natural unemployment rate 自然失业率
    alpha_pi = 0.5
    alpha_u = 1 - alpha_pi
    
    
    #####################################
    ##          Agent模型参数          ##
    #####################################
    gamma = 0.5
    beta = 0.5
    
    pw_low = 0.60
    pw_high = 0.95
    pc_low = 0.1
    pc_high = 0.2
    
    pw_delta = 0.3
    pc_delta = 0.001