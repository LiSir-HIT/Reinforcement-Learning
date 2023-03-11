import numpy as np
import torch
import matplotlib.pyplot as plt
import gym
from parsers import args
from RL_brain import ReplayBuffer, DDPG
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# -------------------------------------- #
# 环境加载
# -------------------------------------- #

env_name = "MountainCarContinuous-v0"  # 连续型动作
env = gym.make(env_name, render_mode="human")
n_states = env.observation_space.shape[0]  # 状态数 2
n_actions = env.action_space.shape[0]  # 动作数 1
action_bound = env.action_space.high[0]  # 动作的最大值 1.0


# -------------------------------------- #
# 模型构建
# -------------------------------------- #

# 经验回放池实例化
replay_buffer = ReplayBuffer(capacity=args.buffer_size)
# 模型实例化
agent = DDPG(n_states = n_states,  # 状态数
             n_hiddens = args.n_hiddens,  # 隐含层数
             n_actions = n_actions,  # 动作数
             action_bound = action_bound,  # 动作最大值
             sigma = args.sigma,  # 高斯噪声
             actor_lr = args.actor_lr,  # 策略网络学习率
             critic_lr = args.critic_lr,  # 价值网络学习率
             tau = args.tau,  # 软更新系数
             gamma = args.gamma,  # 折扣因子
             device = device
            )

# -------------------------------------- #
# 模型训练
# -------------------------------------- #

return_list = []  # 记录每个回合的return
mean_return_list = []  # 记录每个回合的return均值

for i in range(10):  # 迭代10回合
    episode_return = 0  # 累计每条链上的reward
    state = env.reset()[0]  # 初始时的状态
    done = False  # 回合结束标记

    while not done:
        # 获取当前状态对应的动作
        action = agent.take_action(state)
        # 环境更新
        next_state, reward, done, _, _ = env.step(action)
        # 更新经验回放池
        replay_buffer.add(state, action, reward, next_state, done)
        # 状态更新
        state = next_state
        # 累计每一步的reward
        episode_return += reward

        # 如果经验池超过容量，开始训练
        if replay_buffer.size() > args.min_size:
            # 经验池随机采样batch_size组
            s, a, r, ns, d = replay_buffer.sample(args.batch_size)
            # 构造数据集
            transition_dict = {
                'states': s,
                'actions': a,
                'rewards': r,
                'next_states': ns,
                'dones': d,
            }
            # 模型训练
            agent.update(transition_dict)
    
    # 保存每一个回合的回报
    return_list.append(episode_return)
    mean_return_list.append(np.mean(return_list[-10:]))  # 平滑

    # 打印回合信息
    print(f'iter:{i}, return:{episode_return}, mean_return:{np.mean(return_list[-10:])}')

# 关闭动画窗格
env.close()

# -------------------------------------- #
# 绘图
# -------------------------------------- #

x_range = list(range(len(return_list)))

plt.subplot(121)
plt.plot(x_range, return_list)  # 每个回合return
plt.xlabel('episode')
plt.ylabel('return')
plt.subplot(122)
plt.plot(x_range, mean_return_list)  # 每回合return均值
plt.xlabel('episode')
plt.ylabel('mean_return')
