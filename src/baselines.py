'''
baseline control algorithms
'''

# PPO Proximal Policy Optimization
# from stable_baselines3.ppo import PPO
# PPO_model = PPO('MlpPolicy', 'CartPole-v1', verbose=1)

import sys
import os
current_dir = os.path.dirname(__file__) # 获取当前文件的目录
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir)) # 获取上级目录
sys.path.append(parent_dir) # 将上级目录添加到 sys.path


import gymnasium as gym
from copy import deepcopy
from src.step_simulation import step_simulation
from config import Configuration, EconomicCrisisConfig, ReconstructionConfig, EconomicProsperityConfig
from src.agent import agent
from src.firm import firm
from src.bank import bank
from src.utils import init_agents, gini_coefficient
import numpy as np
from gymnasium.utils import seeding

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def reward_func(event:int, log:dict) -> float:
    '''
    reward function for three events
    '''
    last_timestep = max(log.keys()) # get the last timestep
    if event == 1:
        # r_gdp = np.log(log[last_timestep]['GDP']) - np.log(prev_state['GDP'])
        r_gini = (1 - log[last_timestep]['gini']) ** 3 # penalty for high gini coefficient
        r_unem = 1/(1 + np.exp(5*(log[last_timestep]['unemployment_rate'] - 0.05)))  # 失业率超5%时快速衰减
        
        production_history = [log[t]['production'] for t in range(last_timestep-12, last_timestep)]
        capacity_ratio = log[last_timestep]['production'] / (np.mean(production_history) * 1.1)
        r_prod = np.tanh(capacity_ratio - 1)  # 鼓励扩张产能，但超过历史均值10%时饱和

        deposit_std = np.std(log[last_timestep]['wage'])
        deposit_fairness = np.log(1 + log[last_timestep]['avg_wage']) - 0.2 * deposit_std
        r_deposit = 1/(1 + np.exp(-deposit_fairness))  # 鼓励存款公平
        
        # 当基尼系数超过0.4时增强公平权重
        beta = 1.0 + 2.0 * sigmoid(10*(log[last_timestep]['gini'] - 0.4))

        # 当失业率超过自然失业率时提升生产权重
        alpha = 1.0 + 1.5 * (log[last_timestep]['unemployment_rate'] > 0.3)

        reward = alpha * (0.5 * r_unem + 0.5 * r_prod) + beta * (0.5 * r_gini + 0.5 * r_deposit)
        return reward
    else:
        raise ValueError("Invalid event")
from gym.envs.box2d.lunar_lander import LunarLanderContinuous

class MEABM_gym(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    def __init__(self, event: int, step_len: int=50):
        super(MEABM_gym, self).__init__()
        self.action_space = gym.spaces.Box(0, 0.5, (1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        self.timestep = None
        self.event = event
        self.step_len = step_len
        self.log = None
        
        

    def step(self, action):
        '''
        action: []
        '''

        action = np.clip(action, 0, 0.5)
        self.F, self.B, self.agents, log = step_simulation(
            self.config, 
            event=self.event, 
            intervention=True, 
            action=action,
            step=self.timestep, 
            length=self.step_len, 
            firm=deepcopy(self.F), bank=deepcopy(self.B), agents=deepcopy(self.agents), log=deepcopy(self.log))
        self.log = deepcopy(log)
        # print('hhhhhhhhhhhhhh:', self.log[list(self.log.keys())[-1]]['price'])
        self.timestep += self.step_len
        obs = [
            self.F.P, 
            self.B.rate, 
            self.F.G, 
            self.log[list(self.log)[-1]]['imbalance'],
            self.log[list(self.log)[-1]]['inflation_rate'],
            1-sum([a.l for a in self.agents])/self.config.num_agents,
            sum([a.w for a in self.agents])/self.config.num_agents,
            self.log[list(self.log)[-1]]['GDP'],
            gini_coefficient([a.w for a in self.agents]),
            ]
        
        reward = reward_func(self.event, self.log)
        terminated = True if self.timestep == 600 else False
        return obs, reward, terminated, False, self.log

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # if seed is not None:
        #     self._np_random, self._np_random_seed = seeding.np_random(seed)
            
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

        self.log = {
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
            1-sum([a.l for a in self.agents])/self.config.num_agents,  # unemployment rate
            sum([a.w for a in self.agents])/self.config.num_agents,    # avg wage
            0., # GDP
            gini_coefficient([a.w for a in self.agents]),
            ], self.log


if __name__ == '__main__':
    env = MEABM_gym(event=1, step_len=50)
    obs, info = env.reset()
    print(obs)
    
    for i in range(30):
        obs, reward, terminated, truncated, info = env.step(0.2)
        print(i, reward, terminated, 'log len:', len(env.log))
        if terminated:
            break
