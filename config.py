from time import time
class Configuration:
    def __init__(self):
        #####################################
        ##       事 件 与 干 预 参 数       ##
        #####################################
        self.event_flag = False
        self.event_type = 0
        self.event_start = 200
        self.event_end = 400
        self.intervent_start = 200
        self.intervent_end = 600
        
        #####################################
        ##           仿 真 参 数           ##
        #####################################
        self.seed = 124 # int(time()) # 123
        self.num_agents = 100
        self.num_time_steps = 12 * 50
        
        
        #####################################
        ##        企 业 模  型 参 数        ##
        #####################################
        self.A = 1.0                 # 生产率
        self.alpha_w = 0.05          # 工资调整系数
        self.alpha_p = 0.10          # 价格调整系数
        self.alpha_c = 0.05
        self.init_good = 10000       # 初始商品库存
        self.init_cap = 1e7          # 初始资本
        self.k_labor = 0.6           # 柯布道格拉斯函数, 劳动弹性系数
        self.k_capital = 1 - self.k_labor # 柯布道格拉斯函数, 资本弹性系数
        
        #####################################
        ##           银行模型参数           ##
        #####################################
        self.rn = 0.01 # 0.01         # natural interest rate 自然利率
        self.pi_t = 0.05 # 0.02       # target inflation rate 目标通膨率
        self.un = 0.05 # 0.04         # natural unemployment rate 自然失业率
        self.alpha_pi = 1.0
        self.alpha_u = 1.0
        
        self.r_max = 0.20             # maximal interest rate
        self.r_min = -0.05            # minimal interest rate
        self.init_assets = 1e7
        
        #####################################
        ##          Agent模型参数          ##
        #####################################
        self.gamma = 0.5
        self.beta = 0.5
        self.wage_mean = 80
        self.wage_std = 65
        
        self.pw_low = 0.6 # 0.2 # 0.60     # 就业意愿下限
        self.pw_high = 1.0 # 0.4 # 1.0     # 就业意愿上限
        self.pc_low = 0.1 # 0.4 # 0.1
        self.pc_high = 0.5 # 0.8 # 0.4
        
        self.pw_delta = 0.03
        self.pc_delta = 0.01
        
        #####################################
        ##           市场模型参数           ##
        #####################################
        self.tax_rate_good = 0.02   # 向企业征收营业税
        self.communications_num = 1000 # 智能体之间两两沟通的次数
        self.bounded_conf = 0.02 # bounded confidence, opinion差值小于此阈值的才会发生opinion fusion
        self.opinion_fusion_factor = 0.001 # opinion fusion factor
        
        
        #####################################
        ##            GPT 参 数            ##
        #####################################
        self.gpt_model = 'gpt-4o-mini'
        self.max_tokens = 1024
        self.temperature = 0.5

class EconomicCrisisConfig(Configuration):
    '''
    模拟1925-1935的经济危机期
    '''
    def __init__(self):
        super().__init__()
        #####################################
        ##       事 件 与 干 预 参 数       ##
        #####################################
        self.event_type = 1
        self.event_flag = True
        self.event_start = 200
        self.event_end = 400
        self.intervent_start = 200
        self.intervent_end = 400    
        
        #####################################
        ##        企 业 模  型 参 数        ##
        #####################################
        self.k_labor = 0.7
        self.k_capital = 1 - self.k_labor
        
        #####################################
        ##           银行模型参数           ##
        #####################################
        self.rn = 0.05
        self.r_max = 0.3
        
        #####################################
        ##          Agent模型参数          ##
        #####################################
        self.pw_low = 0.45
        self.pw_high = 0.85

class ReconstructionConfig(Configuration):
    '''
    模拟二战后1945-1955的欧洲重建时期
    '''
    def __init__(self):
        super().__init__()
        #####################################
        ##           干 预 参 数           ##
        #####################################
        self.event_type = 2
        
        
        #####################################
        ##        企 业 模  型 参 数        ##
        #####################################
        self.k_labor = 0.6
        self.k_capital = 1 - self.k_labor # 0.4
        
        
        #####################################
        ##           银行模型参数           ##
        #####################################
        self.rn = 0.025
        self.r_max = 0.10
        
        #####################################
        ##          Agent模型参数          ##
        #####################################
        self.pw_low = 0.35
        self.pw_high = 0.45


class EconomicProsperityConfig(Configuration):
    '''
    模拟1960s-2000s的经济繁荣期
    '''
    def __init__(self):
        super().__init__()
        #####################################
        ##           干 预 参 数           ##
        #####################################
        self.event_type = 3
        self.event_start = 50
        self.event_end = 400
        
        #####################################
        ##        企 业 模  型 参 数        ##
        #####################################
        self.k_labor = 0.7
        self.k_capital = 1 - self.k_labor
        
        
        #####################################
        ##           银行模型参数           ##
        #####################################
        self.rn = 0.01
        self.r_max = 0.10
        
        #####################################
        ##          Agent模型参数          ##
        #####################################
        self.pw_low = 0.75
        self.pw_high = 0.95