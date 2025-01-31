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
from src.step_simulation import step_simulation
from config import Configuration, EconomicCrisisConfig, ReconstructionConfig, EconomicProsperityConfig
from src.agent import agent
from src.firm import firm
from src.bank import bank
from src.utils import init_agents
import numpy as np
class MEABM_gym(gym.env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    def __init__(self):
        super(MEABM_gym, self).__init__()
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(4,))
        
        
        

    def step(self, action):
        
        return obs, reward, terminated, None, {}

    def reset(self, event: int):
        if event == 0:
            self.config = Configuration()
        elif event == 1:
            self.config = EconomicCrisisConfig()
        elif event == 2:
            self.config = ReconstructionConfig()
        elif event == 3:
            self.config = EconomicProsperityConfig()
        else:
            raise ValueError("Invalid event")
    
        F0 = firm(A=self.config.A, 
                alpha_w=self.config.alpha_w, 
                alpha_p=self.config.alpha_p,
                alpha_c=self.config.alpha_c,
                init_good=self.config.init_good,
                init_cap=self.config.init_cap,
                k_labor=self.config.k_labor,
                k_capital=self.config.k_capital,
                )
        B0 = bank(rn=self.config.rn, 
                pi_t=self.config.pi_t, 
                un=self.config.un, 
                alpha_pi=self.config.alpha_pi, 
                alpha_u=self.config.alpha_u, 
                num_agents=self.config.num_agents, 
                rate_max=self.config.r_max, 
                rate_min=self.config.r_min,
                init_assets=self.config.init_assets,
                )
        agents0 = init_agents(self.config)
        
        F0.P = np.mean([a.w*a.pc for a in agents0]) # t=0 initial price