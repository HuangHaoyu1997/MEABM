from src.agent import agent

class bank:
    '''
    central bank
    '''
    def __init__(self, rn:float, pi_t:float, un:float, alpha_pi:float, alpha_u:float, num_agents:int):
        self.rn = rn        # natural interest rate, constant value
        self.pi_t = pi_t    # target inflation rate, constant value
        self.un = un        # natural unemployment rate, constant value
        self.rate = rn      # initial interest rate = natural rate
        self.alpha_pi = alpha_pi
        self.alpha_u = alpha_u
        self.deposits = {i:0. for i in range(num_agents)} # 注意，id从0开始编号，可能需要修改
    
    def interest(self, agent_list:list[agent],):
        '''
        interest rate payment anually
        '''
        for a in agent_list:
            self.deposits[a.id] *= (1 + self.rate)
    
    def deposit(self, agent_id, income:float):
        '''
        update the deposit of the agent with agent_id
        '''
        # print('before:', agent_id, self.deposits[agent_id])
        if agent_id in self.deposits:
            self.deposits[agent_id] += income
        else:
            self.deposits[agent_id] = income
        # print('after:', agent_id, self.deposits[agent_id])
    
    def rate_adjustment(self, unemployment_rate:float, inflation_rate:float):
        '''
        Taylor rule for interest rate adjustment
        
        '''
        rate_after = max(self.rn + self.pi_t + self.alpha_pi * (inflation_rate - self.pi_t) + self.alpha_u * (self.un - unemployment_rate), -0.05)
        self.rate = rate_after
        return rate_after