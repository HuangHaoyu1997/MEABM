'''
baseline control algorithms
'''

# PPO Proximal Policy Optimization
# from stable_baselines3.ppo import PPO
# PPO_model = PPO('MlpPolicy', 'CartPole-v1', verbose=1)


from itertools import islice
import pickle
import time
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

def reward_func(event:int, log:dict, step_len:int) -> float:
    '''
    reward function for three events
    '''
    last_timestep = max(log.keys()) # get the last timestep
    print(len(log), last_timestep, last_timestep-step_len)
    if event == 1:
        # r_gdp = np.log(log[last_timestep]['GDP']) - np.log(prev_state['GDP'])
        mean_gini = np.mean([log[t]['gini'] for t in range(last_timestep-step_len, last_timestep)])
        r_gini = (1 - mean_gini) ** 3 # penalty for high gini coefficient

        mean_unem = np.mean([log[t]['unemployment_rate'] for t in range(last_timestep-step_len, last_timestep)])
        r_unem = 1/(1 + np.exp(5*(mean_unem - 0.35)))  # 失业率超5%时快速衰减
        
        
        production_history = [log[t]['production'] for t in range(last_timestep-step_len, last_timestep)]

        # print(log[last_timestep]['production'], np.mean(production_history))

        capacity_ratio = log[last_timestep]['production'] / (np.mean(production_history) * 1.1 + 1e-3)
        r_prod = np.tanh(capacity_ratio - 1)  # 鼓励扩张产能，但超过历史均值10%时饱和

        # deposit_std = np.std(log[last_timestep]['wage'])
        # deposit_fairness = np.log(1 + log[last_timestep]['avg_wage']) - 0.2 * deposit_std
        # r_deposit = 1/(1 + np.exp(-deposit_fairness))  # 鼓励存款公平
        
        # 当基尼系数超过0.4时增强公平权重
        beta = 1.0 + 2.0 * sigmoid(10*(mean_gini - 0.4))

        # 当失业率超过自然失业率时提升生产权重
        alpha = 1.0 + 0.5 * (mean_unem > 0.5)
        # print(alpha, beta, mean_unem, mean_gini)
        reward = alpha * (0.5 * r_unem + 0.5 * r_prod) + beta * r_gini # (0.5 * r_gini + 0.5 * r_deposit)
        return reward
    else:
        raise ValueError("Invalid event")

class MEABM_gym(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    def __init__(self, event: int, step_len: int=50, seed: int=42):
        super(MEABM_gym, self).__init__()
        
        self.timestep = None
        self.event = event
        self.step_len = step_len
        self.log = None
        self.rng = np.random.default_rng(seed)

        if self.event == 0:    self.config = Configuration()
        elif self.event == 1:  self.config = EconomicCrisisConfig()
        elif self.event == 2:  self.config = ReconstructionConfig()
        elif self.event == 3:  self.config = EconomicProsperityConfig()
        else:                  raise ValueError("Invalid event")

        self.action_space = gym.spaces.Box(self.config.r_min, self.config.r_max, (1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        
        

    def step(self, action):
        self.F, self.B, self.agents, log = step_simulation(
                                                            self.config, 
                                                            event=self.event, 
                                                            intervention=True, 
                                                            action=action,
                                                            step=self.timestep, 
                                                            length=self.step_len, 
                                                            firm=deepcopy(self.F), 
                                                            bank=deepcopy(self.B), 
                                                            agents=deepcopy(self.agents), 
                                                            log=deepcopy(self.log),
                                                            )
        
        self.log = deepcopy(log)
        self.timestep += self.step_len
        last_step = list(self.log)[-1]
        obs = [
            np.mean([self.log[t]['price'] for t in range(last_step-self.step_len, last_step)]), # price
            np.mean([self.log[t]['rate'] for t in range(last_step-self.step_len, last_step)]), # interest rate
            np.mean([self.log[t]['production'] for t in range(last_step-self.step_len, last_step)]), # 实际上是储量
            np.mean([self.log[t]['imbalance'] for t in range(last_step-self.step_len, last_step)]), # imbalance
            np.mean([self.log[t]['inflation_rate'] for t in range(last_step-self.step_len, last_step)]), # 通胀率
            np.mean([self.log[t]['unemployment_rate'] for t in range(last_step-self.step_len, last_step)]), # unemployment rate
            np.mean([self.log[t]['avg_wage'] for t in range(last_step-self.step_len, last_step)]), # avg_wage
            np.mean([self.log[t]['GDP'] for t in range(last_step-self.step_len, last_step)]), # GDP
            np.mean([self.log[t]['gini'] for t in range(last_step-self.step_len, last_step)]), # gini
            ]
        
        reward = reward_func(self.event, self.log, self.step_len)
        terminated = True if self.timestep == 600 else False
        return obs, reward, terminated, False, self.log

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # if seed is not None:
        #     self._np_random, self._np_random_seed = seeding.np_random(seed)
            
        self.timestep = 0
    
        self.F = firm(A=self.config.A, 
                alpha_w=self.config.alpha_w, 
                alpha_p=self.config.alpha_p,
                alpha_c=self.config.alpha_c,
                init_good=self.config.init_good,
                init_cap=self.config.init_cap,
                k_labor=self.config.k_labor,
                k_capital=self.config.k_capital,
                rng=self.rng,
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
        self.agents = init_agents(self.config, self.rng)
        
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
                'inventory': self.F.G,
                'production': 0.,
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
            1 - sum([a.l for a in self.agents]) / self.config.num_agents,  # unemployment rate
            sum([a.w for a in self.agents]) / self.config.num_agents,    # avg wage
            0., # GDP
            gini_coefficient([a.w for a in self.agents]),
            ], self.log


if __name__ == '__main__':
    env = MEABM_gym(event=1, step_len=50)
    env.reset()
    print(env.action_space, env.observation_space)
    
    
    import gymnasium as gym


    def make_env(seed):
        def thunk():
            env = MEABM_gym(event=1, step_len=50, seed=seed)
            # env = gym.make(env_id)
            # env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.ClipAction(env)
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), None)
            # 可选：对奖励归一化和裁剪（此处未使用）
            env.reset(seed=seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk



    envs = gym.vector.AsyncVectorEnv( # AsyncVectorEnv( SyncVectorEnv
        [make_env(123+i) for i in range(8)]
    )
    print(envs.single_observation_space.shape)
    # print(envs.action_space.sample().shape)
    obs, reward, terminated, truncated, info = env.step(0.2)
    # obs, reward, terminated, truncated, info = env.step([0.2])
    # print(obs, reward, terminated, truncated, len(info))
    obs, info = envs.reset()
    obs, reward, terminated, truncated, info = envs.step([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
    print(len(obs), reward, terminated, truncated, len(info))

    
    obs, info = env.reset()
    for i in range(30):
        obs, reward, terminated, truncated, info = env.step(0.05)
        if terminated:
            break
    
    with open('log.pkl', 'wb') as f:
        pickle.dump(info, f)

    # subset_info = dict(islice(info.items(), 550, 601))
    # reward = reward_func(event=1, log=subset_info, step_len=50)
    # print(reward)


