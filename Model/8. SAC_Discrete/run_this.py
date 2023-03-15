import gym 
import torch
import numpy as np
import matplotlib.pyplot as plt
from RL_brain import ReplayBuffer, SAC

# -------------------------------------- #
# 参数设置
# -------------------------------------- #

num_epochs = 1  # 训练回合数
capacity = 500  # 经验池容量
min_size = 200 # 经验池训练容量
batch_size = 64
n_hiddens = 64
actor_lr = 1e-3  # 策略网络学习率
critic_lr = 1e-2  # 价值网络学习率
alpha_lr = 1e-2  # 课训练变量的学习率
target_entropy = -1
tau = 0.005  # 软更新参数
gamma = 0.9  # 折扣因子
device = torch.device('cuda') if torch.cuda.is_available() \
                            else torch.device('cpu')

# -------------------------------------- #
# 环境加载
# -------------------------------------- #

env_name = "CartPole-v1"
env = gym.make(env_name, render_mode="human")
n_states = env.observation_space.shape[0]  # 状态数 4
n_actions = env.action_space.n  # 动作数 2

# -------------------------------------- #
# 模型构建
# -------------------------------------- #

agent = SAC(n_states = n_states,
            n_hiddens = n_hiddens,
            n_actions = n_actions,
            actor_lr = actor_lr,
            critic_lr = critic_lr,
            alpha_lr = alpha_lr,
            target_entropy = target_entropy,
            tau = tau,
            gamma = gamma,
            device = device,
            )

# -------------------------------------- #
# 经验回放池
# -------------------------------------- #

buffer = ReplayBuffer(capacity=capacity)

# -------------------------------------- #
# 模型构建
# -------------------------------------- #

return_list = []  # 保存每回合的return

for i in range(num_epochs):
    state = env.reset()[0]
    epochs_return = 0  # 累计每个时刻的reward
    done = False  # 回合结束标志

    while not done:
        # 动作选择
        action = agent.take_action(state)
        # 环境更新
        next_state, reward, done, _, _ = env.step(action)
        # 将数据添加到经验池
        buffer.add(state, action, reward, next_state, done)
        # 状态更新
        state = next_state
        # 累计回合奖励
        epochs_return += reward

        # 经验池超过要求容量，就开始训练
        if buffer.size() > min_size:
            s, a, r, ns, d = buffer.sample(batch_size)  # 每次取出batch组数据
            # 构造数据集
            transition_dict = {'states': s,
                               'actions': a,
                               'rewards': r,
                               'next_states': ns,
                               'dones': d}
            # 模型训练
            agent.update(transition_dict)
    # 保存每个回合return
    return_list.append(epochs_return)
    
    # 打印回合信息
    print(f'iter:{i}, return:{np.mean(return_list[-10:])}')

# -------------------------------------- #
# 绘图
# -------------------------------------- #

plt.plot(return_list)
plt.title('return')
plt.show()
