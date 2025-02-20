from src.baselines import MEABM_gym
from src.utils import plot_log

if __name__ == '__main__':
    env = MEABM_gym(event=1, step_len=50, seed=42)
    
    
    for i in range(5):
        rewards = 0
        obs, info = env.reset(seed=42+i)
        done = False
        while not done:
            action = 0.5
            obs, reward, done, trun, info = env.step(action)
            rewards += reward
        print(rewards)
        plot_log(img_name=f'./figs/MPC_exp/event1_step50_action_0.5_seed42_{i}.png', log=info, config=env.config, save=True)