'''
baseline control algorithms
'''

# PPO Proximal Policy Optimization
from stable_baselines3.ppo import PPO
PPO_model = PPO('MlpPolicy', 'CartPole-v1', verbose=1)

import sys
import os
current_dir = os.path.dirname(__file__) # 获取当前文件的目录
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir)) # 获取上级目录
sys.path.append(parent_dir) # 将上级目录添加到 sys.path


import gym
from copy import deepcopy
from src.step_simulation import step_simulation
from config import Configuration, EconomicCrisisConfig, ReconstructionConfig, EconomicProsperityConfig
from src.agent import agent
from src.firm import firm
from src.bank import bank
from src.utils import init_agents, gini_coefficient
import numpy as np


def reward_func(event:int, log:dict):
    '''
    reward function for three events
    '''
    if event == 1:
        


class MEABM_gym(gym.env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    def __init__(self, event: int):
        super(MEABM_gym, self).__init__()
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=10, shape=(8,))
        self.timestep = None
        self.event = event
        
        

    def step(self, action, log):
        '''
        action: []
        '''
        self.F, self.B, self.agents, log = step_simulation(
            self.config, 
            event=self.event, 
            intervention=True, 
            step=self.timestep, 
            length=20, 
            firm=deepcopy(self.F), bank=deepcopy(self.B), agents=deepcopy(self.agents), log=deepcopy(log))
        self.timestep += 20
        obs = [
            self.F.P, 
            self.B.rate, 
            self.F.G, 
            0., 
            0.,
            1-sum([a.l for a in self.agents])/self.config.num_agents,
            sum([a.w for a in self.agents])/self.config.num_agents,
            gini_coefficient([a.w for a in self.agents]),
            ]
        
        reward = 
        terminated = True if self.timestep == 600 else False
        return obs, reward, terminated, None, log

    def reset(self):
        
        self.timestep = 0
        if self.event == 0:    self.config = Configuration()
        elif self.event == 1:  self.config = EconomicCrisisConfig()
        elif self.event == 2:  self.config = ReconstructionConfig()
        elif self.event == 3:  self.config = EconomicProsperityConfig()
        else:                  raise ValueError("Invalid event")
    
        self.F = firm(A=self.config.A, 
                alpha_w=self.config.alpha_w, 
                alpha_p=self.config.alpha_p,
                alpha_c=self.config.alpha_c,
                init_good=self.config.init_good,
                init_cap=self.config.init_cap,
                k_labor=self.config.k_labor,
                k_capital=self.config.k_capital,
                )
        self.B = bank(rn=self.config.rn, 
                pi_t=self.config.pi_t, 
                un=self.config.un, 
                alpha_pi=self.config.alpha_pi, 
                alpha_u=self.config.alpha_u, 
                num_agents=self.config.num_agents, 
                rate_max=self.config.r_max, 
                rate_min=self.config.r_min,
                init_assets=self.config.init_assets,
                )
        self.agents = init_agents(self.config)
        
        self.F.P = np.mean([a.w*a.pc for a in self.agents])*5 # t=0 initial price

        log = {
            0:{
                'year': 0,
                'work_state': [a.l for a in self.agents], 
                'pw': [a.pw for a in self.agents],
                'pc': [a.pc for a in self.agents],
                'wage': [a.w for a in self.agents],
                'price': self.F.P, 
                'rate': self.B.rate, 
                'production': self.F.G,
                'imbalance': 0.,
                'inflation_rate': 0.,
                'taxes': 0,
                'unemployment_rate': 1-sum([a.l for a in self.agents])/self.config.num_agents,
                'deposit': {i: 0.0 for i in range(self.config.num_agents)},
                'avg_wage': sum([a.w for a in self.agents])/self.config.num_agents,
                'GDP': 0.,
                'capital': self.F.capital,
                'assets': self.B.assets,
                'gini': gini_coefficient([a.w for a in self.agents]),
                }
            }

        return [
            self.F.P, 
            self.B.rate, 
            self.F.G, 
            0., 
            0.,
            1-sum([a.l for a in self.agents])/self.config.num_agents,
            sum([a.w for a in self.agents])/self.config.num_agents,
            gini_coefficient([a.w for a in self.agents]),
            ], log