import gym
import numpy as np
import matplotlib.pyplot as plt
from RL_brain import PolicyGradient

# ------------------------------- #
# 模型参数设置
# ------------------------------- #

n_hiddens = 16  # 隐含层个数
learning_rate = 2e-3  # 学习率
gamma = 0.9  # 折扣因子
return_list = []  # 保存每回合的reward
max_q_value = 0  # 初始的动作价值函数
max_q_value_list = []  # 保存每一step的动作价值函数

# ------------------------------- #
#（1）加载环境
# ------------------------------- #

# 连续性动作
env = gym.make("CartPole-v1", render_mode="human")
n_states = env.observation_space.shape[0]  # 状态数 4
n_actions = env.action_space.n  # 动作数 2

# ------------------------------- #
#（2）模型实例化
# ------------------------------- #

agent = PolicyGradient(n_states=n_states,  # 4
                       n_hiddens=n_hiddens,  # 16
                       n_actions=n_actions,  # 2
                       learning_rate=learning_rate,  # 学习率
                       gamma=gamma)  # 折扣因子

# ------------------------------- #
#（3）训练
# ------------------------------- #

for i in range(100):  # 训练10回合
    # 记录每个回合的return
    episode_return = 0
    # 存放状态
    transition_dict = {
        'states': [],
        'actions': [],
        'next_states': [],
        'rewards': [],
        'dones': [],
    }
    # 获取初始状态
    state = env.reset()[0]
    # 结束的标记
    done = False

    # 开始迭代
    while not done:
        # 动作选择
        action = agent.take_action(state)  # 对某一状态采取动作
        # 动作价值函数，曲线平滑
        max_q_value = agent.max_q_value(state) * 0.005 + max_q_value * 0.995
        # 保存每一step的动作价值函数
        max_q_value_list.append(max_q_value)
        # 环境更新
        next_state, reward, done, _, _ = env.step(action)
        # 保存每个回合的所有信息
        transition_dict['states'].append(state)
        transition_dict['actions'].append(action)
        transition_dict['next_states'].append(next_state)
        transition_dict['rewards'].append(reward)
        transition_dict['dones'].append(done)
        # 状态更新
        state = next_state
        # 记录每个回合的return
        episode_return += reward

    # 保存每个回合的return
    return_list.append(episode_return)
    # 一整个回合走完了再训练模型
    agent.learn(transition_dict)

    # 打印回合信息
    print(f'iter:{i}, return:{np.mean(return_list[-10:])}')

# 关闭动画
env.close()
        
# -------------------------------------- #
# 绘图
# -------------------------------------- #

plt.subplot(121)
plt.plot(return_list)
plt.title('return')
plt.subplot(122)
plt.plot(max_q_value_list)
plt.title('max_q_value')
plt.show()
