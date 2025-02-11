from src.agent import agent

class bank:
    '''
    central bank
    '''
    def __init__(self, 
                 rn:float, 
                 pi_t:float, 
                 un:float, 
                 alpha_pi:float, 
                 alpha_u:float, 
                 num_agents:int, 
                 rate_max:float,
                 rate_min:float,
                 init_assets:float,
                 ):
        
        self.natural_rate = rn        # natural interest rate, constant value
        self.pi_t = pi_t              # target inflation rate, constant value
        self.un = un                  # natural unemployment rate, constant value
        self.rate = rn + pi_t         # initial interest rate = natural rate
        self.alpha_pi = alpha_pi
        self.alpha_u = alpha_u
        self.rate_min = rate_min      # minimal interest rate
        self.rate_max = rate_max
        self.assets = init_assets
        self.deposits = {i:0. for i in range(num_agents)} # 注意，id从0开始编号，可能需要修改
    
    def interest(self, agent_list:list[agent],):
        '''
        interest rate payment annually
        '''
        for a in agent_list:
            self.assets -= self.deposits[a.id] * self.rate
            self.deposits[a.id] = self.deposits[a.id] * (1 + self.rate)
    
    def deposit(self, agent_id, income:float):
        '''
        update the deposit of the agent with agent_id
        '''
        # print('before:', agent_id, self.deposits[agent_id])
        if agent_id in self.deposits:
            self.deposits[agent_id] += income
        else:
            self.deposits[agent_id] = income
        # print('after:', agent_id, income, type(self.deposits[agent_id]), self.deposits[agent_id])
    
    def rate_adjustment(self, unemployment_rate:float, inflation_rate:float):
        '''
        Taylor rule for interest rate adjustment
        若 inflation rate > pi_t, 通胀高于预期, 意味着需求过旺, 超出了经济的生产能力, 从而推高了物价.
        提高利率会增加借贷成本, 减少消费和投资需求, 从而抑制整体需求. 这种需求抑制可以帮助减缓通胀压力, 使通胀率逐步回到目标水平.
        
        若 unemployment rate < un, 实际失业率低于自然失业率, 说明劳动力市场需求旺盛, 几乎所有愿意工作的人都能找到工作.
        这种情况下, 企业可能会难以找到足够的劳动力, 从而增加工资支出以吸引员工. 
        更高的工资意味着更高的生产成本, 这些成本往往会转嫁给消费者, 导致商品和服务价格上涨. 
        持续的工资和价格上涨可能会导致经济过热, 形成通胀压力. 提高名义利率可以减缓经济活动, 防止经济从过热走向失控.
        '''
        rate_after = min(
            max(self.natural_rate + self.pi_t + \
                self.alpha_pi * (inflation_rate - self.pi_t) + \
                    self.alpha_u * (self.un - unemployment_rate), 
                    self.rate_min), 
            self.rate_max)
        self.rate = rate_after
        return rate_after