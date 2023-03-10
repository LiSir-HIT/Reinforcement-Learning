import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
from RL_brain import ActorCritic

# ----------------------------------------- #
# 参数设置
# ----------------------------------------- #

num_episodes = 100  # 总迭代次数
gamma = 0.9  # 折扣因子
actor_lr = 1e-3  # 策略网络的学习率
critic_lr = 1e-2  # 价值网络的学习率
n_hiddens = 16  # 隐含层神经元个数
env_name = 'CartPole-v1'
return_list = []  # 保存每个回合的return

# ----------------------------------------- #
# 环境加载
# ----------------------------------------- #

env = gym.make(env_name, render_mode="human")
n_states = env.observation_space.shape[0]  # 状态数 4
n_actions = env.action_space.n  # 动作数 2

# ----------------------------------------- #
# 模型构建
# ----------------------------------------- #

agent = ActorCritic(n_states=n_states,  # 状态数
                    n_hiddens=n_hiddens,  # 隐含层数
                    n_actions=n_actions,  # 动作数
                    actor_lr=actor_lr,  # 策略网络学习率
                    critic_lr=critic_lr,  # 价值网络学习率
                    gamma=gamma)  # 折扣因子

# ----------------------------------------- #
# 训练--回合更新
# ----------------------------------------- #

for i in range(num_episodes):
    
    state = env.reset()[0]  # 环境重置
    done = False  # 任务完成的标记
    episode_return = 0  # 累计每回合的reward

    # 构造数据集，保存每个回合的状态数据
    transition_dict = {
        'states': [],
        'actions': [],
        'next_states': [],
        'rewards': [],
        'dones': [],
    }

    while not done:
        action = agent.take_action(state)  # 动作选择
        next_state, reward, done, _, _  = env.step(action)  # 环境更新
        # 保存每个时刻的状态\动作\...
        transition_dict['states'].append(state)
        transition_dict['actions'].append(action)
        transition_dict['next_states'].append(next_state)
        transition_dict['rewards'].append(reward)
        transition_dict['dones'].append(done)
        # 更新状态
        state = next_state
        # 累计回合奖励
        episode_return += reward
    
    # 保存每个回合的return
    return_list.append(episode_return)
    # 模型训练
    agent.update(transition_dict)

    # 打印回合信息
    print(f'iter:{i}, return:{np.mean(return_list[-10:])}')

# -------------------------------------- #
# 绘图
# -------------------------------------- #

plt.plot(return_list)
plt.title('return')
plt.show()
