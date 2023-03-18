import numpy as np
import matplotlib.pyplot as plt
import torch
from ma_gym.envs.combat.combat import Combat
from RL_brain import PPO
import time

# ----------------------------------------- #
# 参数设置
# ----------------------------------------- #

n_hiddens = 64  # 隐含层数量
actor_lr = 3e-4
critic_lr = 1e-3
gamma = 0.9
lmbda = 0.97
eps = 0.2
device = torch.device('cuda') if torch.cuda.is_available() \
                            else torch.device('cpu')
num_episodes = 10  # 回合数
team_size = 2  # 智能体数量
grid_size = (15, 15)

# ----------------------------------------- #
# 环境设置--onpolicy
# ----------------------------------------- #

# 创建Combat环境，格子世界的大小为15x15，己方智能体和敌方智能体数量都为2
env = Combat(grid_shape=grid_size, n_agents=team_size, n_opponents=team_size)
n_states = env.observation_space[0].shape[0]  # 状态数
n_actions = env.action_space[0].n  # 动作数

# 两个智能体共享同一个策略
agent = PPO(n_states = n_states,
            n_hiddens = n_hiddens,
            n_actions = n_actions,
            actor_lr = actor_lr,
            critic_lr = critic_lr,
            lmbda = lmbda,
            eps = eps,
            gamma = gamma,
            device = device,
            )

# ----------------------------------------- #
# 模型训练
# ----------------------------------------- #

for i in range(num_episodes):
    # 每回合开始前初始化两支队伍的数据集
    transition_dict_1 = {
        'states': [],
        'actions': [],
        'next_states': [],
        'rewards': [],
        'dones': [],
    }
    transition_dict_2 = {
        'states': [],
        'actions': [],
        'next_states': [],
        'rewards': [],
        'dones': [],
    }

    s = env.reset()  # 状态初始化
    terminal = False  # 结束标记

    while not terminal:

        env.render()

        # 动作选择
        a_1 = agent.take_action(s[0])
        a_2 = agent.take_action(s[1])

        # 环境更新
        next_s, r, done, info = env.step([a_1, a_2])

        # 构造数据集
        transition_dict_1['states'].append(s[0])
        transition_dict_1['actions'].append(a_1)
        transition_dict_1['next_states'].append(next_s[0])
        transition_dict_1['dones'].append(False)
        transition_dict_1['rewards'].append(r[0])

        transition_dict_2['states'].append(s[1])
        transition_dict_2['actions'].append(a_2)
        transition_dict_2['next_states'].append(next_s[1])
        transition_dict_2['dones'].append(False)
        transition_dict_2['rewards'].append(r[1])

        s = next_s  # 状态更新
        terminal = all(done)  # 判断当前回合是否都为True，是返回True，不是返回False

        time.sleep(0.1)
    
    print('epoch:', i)

    # 回合训练
    agent.update(transition_dict_1)
    agent.update(transition_dict_2)
