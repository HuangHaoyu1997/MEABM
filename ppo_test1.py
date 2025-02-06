import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import gymnasium as gym
import numpy as np
import torch.nn.functional as F
from torch.distributions import Beta


torch.set_default_dtype(torch.float32)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()

        self.fc1 = layer_init(nn.Linear(state_dim, 64))
        self.fc2 = layer_init(nn.Linear(64, 64))
        self.critic = layer_init(nn.Linear(64, 1), std=1.0)
        self.actor_A = layer_init(nn.Linear(64, action_dim), std=0.01)
        self.actor_B = layer_init(nn.Linear(64, action_dim), std=0.01)

    def get_value(self, x):
        return self.critic(torch.tanh(self.fc2(torch.tanh(self.fc1(x)))))

    def forward(self):
        raise NotImplementedError
    
    def act(self, state, memory):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        alpha = F.softplus(self.actor_A(x))
        beta = F.softplus(self.actor_B(x))
        
        dist = Beta(alpha, beta)
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=1)

        # action_mean = self.actor(state)
        # cov_mat = torch.diag(self.action_var).to(device)
        
        # dist = MultivariateNormal(action_mean, cov_mat)
        # action = dist.sample()
        # action_logprob = dist.log_prob(action)
        
        memory.states.append(state.float())
        memory.actions.append(action.float())
        memory.logprobs.append(action_logprob.float())
        
        return action.detach()
    
    def evaluate(self, state, action):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        alpha = F.softplus(self.actor_A(x))
        beta = F.softplus(self.actor_B(x))
        
        dist = Beta(alpha, beta)
        action = dist.sample()

        # action_mean = torch.squeeze(self.actor(state)).float()
        # action_var = self.action_var.expand_as(action_mean).float()
        # cov_mat = torch.diag_embed(action_var).to(device).float()
        
        # dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(torch.squeeze(action)).sum(dim=1)
        dist_entropy = dist.entropy().sum(dim=1)
        state_value = self.get_value(state)
        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        
        self.policy_old = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.policy_old.act(state, memory).cpu().data.numpy().flatten()
    
    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states).to(device)).detach().float()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device)).detach().float()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs)).to(device).detach().float()
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())
            

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())


def main():
    ############## Hyperparameters ##############
    env_name = ["BipedalWalker-v3", 'Ant-v5', 'InvertedPendulum-v5', "HalfCheetah-v5", "LunarLanderContinuous-v3"]
    render = False
    solved_reward = 300         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = 2000        # max training episodes
    max_timesteps = 500        # max timesteps in one episode
    
    update_timestep = 4000      # update policy every n timesteps
    action_std = 0.5            # constant std for action distribution (Multivariate Normal)
    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor
    
    lr = 3e-4                 # parameters for Adam optimizer
    betas = (0.9, 0.999)
    
    random_seed = 123
    #############################################
    
    # creating environment
    env = gym.make(env_name[0])
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), None)
    env = gym.wrappers.NormalizeReward(env)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print('obs dim:', state_dim, 'act dim:', action_dim)
    
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.reset(seed=random_seed)
        np.random.seed(random_seed)
        env.action_space.seed(random_seed)
        env.observation_space.seed(random_seed)
        torch.backends.cudnn.deterministic = True
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    print('learning rate:', lr, 'beta:', betas)
    
    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0
    
    # training loop
    for i_episode in range(1, max_episodes+1):
        state, info = env.reset()
        for t in range(max_timesteps):
            time_step += 1
            # Running policy_old:
            action = ppo.select_action(state, memory)
            state, reward, done, truncated, info = env.step(action)
            
            # Saving reward and is_terminals:
            # print(reward, reward)
            memory.rewards.append(float(reward))
            memory.is_terminals.append(done)
            
            # update if its time
            if time_step % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0
            running_reward += reward
            if render:
                env.render()
            if done:
                break
        
        avg_length += t
        
        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval*solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './PPO_continuous_solved_{}.pth'.format(env_name))
            break
        
        # save every 500 episodes
        if i_episode % 500 == 0:
            torch.save(ppo.policy.state_dict(), './PPO_continuous_{}.pth'.format(env_name))
            
        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = np.round(running_reward/log_interval, 3)
            
            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0
        

if __name__ == '__main__':
    main()  