from time import time
class Configuration:
    #####################################
    ##           干 预 参 数           ##
    #####################################
    event_start = 200
    event_end = 400
    intervent_start = 200
    intervent_end = 600
    
    #####################################
    ##           仿 真 参 数           ##
    #####################################
    seed = 124 # int(time()) # 123
    num_agents = 100
    num_time_steps = 12 * 50
    
    
    #####################################
    ##           企业模型参数           ##
    #####################################
    
    A = 1.0                 # 生产率
    alpha_w = 0.05          # 工资调整系数
    alpha_p = 0.10          # 价格调整系数
    alpha_c = 0.05
    init_good = 10000       # 初始商品库存
    init_cap = 1e7          # 初始资本
    k_labor = 0.6           # 柯布道格拉斯函数, 劳动弹性系数
    k_capital = 1 - k_labor # 柯布道格拉斯函数, 资本弹性系数
    
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
    wage_std = 65
    
    pw_low = 0.6 # 0.2 # 0.60
    pw_high = 1.0 # 0.4 # 1.0
    pc_low = 0.1 # 0.4 # 0.1
    pc_high = 0.5 # 0.8 # 0.4
    
    pw_delta = 0.03
    pc_delta = 0.01
    
    #####################################
    ##           市场模型参数           ##
    #####################################
    tax_rate_good = 0.02
    
    
    #####################################
    ##            GPT  参数            ##
    #####################################
    gpt_model = 'gpt-4o-mini'
    max_tokens = 1024
    temperature = 0.5