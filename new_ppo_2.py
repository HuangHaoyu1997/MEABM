import os, random, time, datetime
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Beta
import matplotlib.pyplot as plt

class Args:
    exp_name = os.path.basename(__file__).rstrip(".py")  # 实验名称（当前文件名）
    seed = 123
    torch_deterministic = True
    cuda = True
    env_id = "BipedalWalker-v3" # 'Ant-v5' # 'InvertedPendulum-v5' # "HalfCheetah-v5"  # 'BipedalWalker-v3' # "LunarLanderContinuous-v3" # 
    total_episodes = 10000  # 总共训练回合数
    learning_rate = 3e-4
    num_envs = 1    # 这里只使用1个环境
    num_steps = 1024  # 每个episode最大rollout步数（当episode未提前结束时，rollout的数据条数即为num_steps）
    max_epi_len = 500  # 最大episode长度（超过此长度则视为episode结束）
    anneal_lr = True
    gae = True
    gae_lambda = 0.9 # 0.95
    gamma = 0.99
    num_minibatches = 16
    update_epochs = 5  # 每个episode更新时，用采样数据更新的轮数
    norm_adv = True
    clip_coef = 0.1 # 0.2
    clip_vloss = True 
    ent_coef = 0.02   # 熵系数
    vf_coef = 0.2    # 价值函数损失系数
    max_grad_norm = 0.5  # 梯度裁剪最大范数
    target_kl = None   # 目标KL散度（不启用时为None）
    # batch_size在每个回合中根据实际rollout步数确定（<=num_steps）
    minibatch_size = None  # 后续更新时会在每个episode中根据batch_size计算


def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id)
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

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)  # 正交初始化权重
    torch.nn.init.constant_(layer.bias, bias_const)  # 常数初始化偏置
    return layer

class Agent(nn.Module):
    def __init__(self, env: gym.Env):
        super().__init__()
        obs_dim = np.prod(env.observation_space.shape)
        action_dim = np.prod(env.action_space.shape)
        self.fc1 = layer_init(nn.Linear(obs_dim, 64))
        self.fc2 = layer_init(nn.Linear(64, 64))
        self.critic = layer_init(nn.Linear(64, 1), std=1.0)
        self.actor_A = layer_init(nn.Linear(64, action_dim), std=0.01)
        self.actor_B = layer_init(nn.Linear(64, action_dim), std=0.01)
    
    def get_value(self, x):
        # x: (batch, obs_dim)
        x = torch.tanh(self.fc1(x))  # 输出： (batch, 128)
        x = torch.tanh(self.fc2(x))  # 输出： (batch, 64)
        x = self.critic(x)  # 输出： (batch, 1)
        return x

    def get_action_and_value(self, xx, action=None):
        # xx: (batch, obs_dim)
        x = torch.tanh(self.fc1(xx))   # (batch, 128)
        x = torch.tanh(self.fc2(x))      # (batch, 64)
        alpha = F.softplus(self.actor_A(x))  # (batch, action_dim) 保证为正值
        beta = F.softplus(self.actor_B(x))   # (batch, action_dim)
        probs = Beta(alpha, beta)
        if action is None:
            action = probs.sample()  # 从Beta分布采样动作，shape：(batch, action_dim)
        # 返回：动作，动作对应对数概率（求和后为(batch,)），分布的熵（求和后为(batch,)），及状态价值 (batch, 1)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.get_value(xx)


if __name__ == "__main__":
    args = Args()
    date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{date}"
    print(run_name)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env = make_env(args.env_id, args.seed)()
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"
    agent = Agent(env).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    plt.ion()
    fig, ax = plt.subplots() 
    ax.set_xlabel("Episode") 
    ax.set_ylabel("Cumulative Reward") 
    ax.set_title("Training Trend")
    ax.grid()
    global_step = 0
    episodes = 0
    all_rewards = []  # 存放每个episode的累计奖励
    mem_obs = []
    mem_actions = []
    mem_logprobs = []
    mem_rewards = []
    mem_dones = []
    mem_values = []

    start_time = time.time()

    # 主训练循环
    # 每个episode更新结束后进行学习更新。这里每个episode收集的数据量最多为args.num_steps（500），
    # 若episode中途done，则rollout数据实际长度<args.num_steps
    for episode in range(1, args.total_episodes + 1):
        # 每个episode开始前reset环境，获取初始观察
        obs = []         # 存储当前episode的观察
        actions = []     # 存储动作
        logprobs = []    # 存储动作对数概率
        rewards = []     # 存储奖励
        dones = []       # 存储done标志（0与1）
        values = []      # 存储状态价值

        next_obs, info = env.reset(seed=args.seed + episode)
        next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)  # shape与env.observation_space一致
        next_done = torch.tensor(0.0).to(device)  # 初始化done标志为0

        epi_reward = 0  # 累计本回合奖励
        # 每个episode至多展开args.num_steps步
        for step in range(args.max_epi_len):
            global_step += 1  # 更新全局步数
            
            obs.append(next_obs)   # 记录当前观察（Tensor，shape: env.observation_space.shape）
            dones.append(next_done)  # 当前done标记

            # 根据当前状态采样动作以及计算对数概率和状态价值
            with torch.no_grad():
                # next_obs.unsqueeze(0)将观察转换为batch形式：(1, obs_dim)
                action, logprob, _, value = agent.get_action_and_value(next_obs.unsqueeze(0))
                # value shape：(1,1)
                values.append(value.flatten())  # flatten后 shape：(1,)
            actions.append(action.squeeze(0))   # 移除batch维度，shape：(action_dim,)
            logprobs.append(logprob.squeeze(0))   # 标量

            # 执行动作。注意：若需要映射动作范围，可在此处调整，例如action_np = action.squeeze(0).cpu().numpy() * 2 - 1
            action_np = action.squeeze(0).cpu().numpy() * 2 - 1 
            next_obs_, reward, done, trun, info = env.step(action_np)
            epi_reward += reward

            rewards.append(torch.tensor(reward, dtype=torch.float32).to(device))
            next_obs = torch.tensor(next_obs_, dtype=torch.float32).to(device)
            next_done = torch.tensor(float(done), dtype=torch.float32).to(device)

            if done: # 如果环境返回done则提前退出rollout
                print(f"Episode {episode} Step {step} Reward: {epi_reward:.2f} Time: {time.time() - start_time:.2f}s")
                start_time = time.time()
                break
        all_rewards.append(epi_reward)
        mem_obs.extend(obs)
        mem_actions.extend(actions)
        mem_logprobs.extend(logprobs)
        mem_rewards.extend(rewards)
        mem_dones.extend(dones)
        mem_values.extend(values)
        
        
        ax.clear()
        ax.plot(all_rewards, label='Cumulative Reward')
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative Reward")
        ax.set_title("Training Trend")
        ax.grid()
        ax.legend()
        plt.pause(0.001)

        if len(mem_obs) >= args.num_steps:
            print('start update')
            # 使用全局记忆池数据构造 batch（注意：此处 batch_size = len(mem_obs) 可能大于 args.num_steps）
            batch_size = len(mem_obs)
            b_obs = torch.stack(mem_obs)
            b_actions = torch.stack(mem_actions)
            b_logprobs = torch.stack(mem_logprobs).reshape(-1)
            b_rewards = torch.stack(mem_rewards).reshape(-1)
            b_dones = torch.stack(mem_dones).reshape(-1)
            b_values = torch.stack(mem_values).reshape(-1)

            # 如果episode提前done，则无后续状态value作引导；因此bootstrap value为0，
            # 若episoderollout用尽且未done，则仍利用下一个状态的价值函数进行引导
            with torch.no_grad():
                if float(next_done.item()) == 1.0:
                    next_value = 0.0
                else:
                    next_value = agent.get_value(next_obs.unsqueeze(0)).reshape(1).item()
        
            # 计算优势和returns，此处使用GAE（广义优势估计）
            if args.gae:
                advantages = torch.zeros(batch_size, device=device)
                lastgaelam = 0.0
                for t in reversed(range(batch_size)):
                    if t == batch_size - 1:
                        nextnonterminal = 1.0 - float(next_done.item())
                        next_values = next_value
                    else:
                        nextnonterminal = 1.0 - b_dones[t+1].item()
                        next_values = b_values[t+1].item()
                    delta = b_rewards[t].item() + args.gamma * next_values * nextnonterminal - b_values[t].item()
                    lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    advantages[t] = lastgaelam
                returns = advantages + b_values
            else:
                # 若不使用GAE，直接计算时间差回报
                returns = torch.zeros(batch_size, device=device)
                for t in reversed(range(batch_size)):
                    if t == batch_size - 1:
                        nextnonterminal = 1.0 - float(next_done.item())
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - b_dones[t+1].item()
                        next_return = returns[t+1].item()
                    returns[t] = b_rewards[t].item() + args.gamma * nextnonterminal * next_return
                advantages = returns - b_values

            # 根据当前数据条数计算小批量大小（向下取整，保证至少有1个样本）
            args.minibatch_size = max(1, batch_size // args.num_minibatches)
        
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1) # b_values已经为1维
                        
            # 学习率退火（每个episode更新一次）
            if args.anneal_lr:
                frac = 1.0 - (episode - 1) / args.total_episodes
                lr_now = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lr_now

        
            # PPO 更新（多轮遍历 mini-batch）
            b_inds = np.arange(batch_size)
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]
                    # 根据当前小批量索引，从数据中取样：注意b_obs shape:(batch_size, obs_dim)
                    # b_actions shape:(batch_size, action_dim)，b_logprobs:(batch_size,), etc.
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    # newlogprob: (minibatch_size,), newvalue: (minibatch_size, 1)
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                    # 此处记录裁剪比例，也可用作监控
                    clipfracs = ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std(unbiased=False) + 1e-8)

                    # 策略梯度损失，采用裁剪目标函数
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # 价值函数损失
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                    # 可选：如果定义了target KL且超过阈值，则可跳出更新循环
                    if args.target_kl is not None and approx_kl > args.target_kl:
                        break
            # 更新完成后清空全局记忆池
            mem_obs.clear()
            mem_actions.clear()
            mem_logprobs.clear()
            mem_rewards.clear()
            mem_dones.clear()
            mem_values.clear()

    env.close()
    plt.ioff() # 关闭交互模式 
    plt.show() # 保持图形窗口