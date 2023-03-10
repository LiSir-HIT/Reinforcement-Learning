import gym
from RL_DQN import DQN, ReplayBuffer
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

# GPU运算
device = torch.device("cuda") if torch.cuda.is_available() \
        else torch.device("cpu")

# ------------------------------- #
# 全局变量
# ------------------------------- #

capacity = 500  # 经验池容量
lr = 2e-3  # 学习率
gamma = 0.9  # 折扣因子
epsilon = 0.9  # 贪心系数
target_update = 200  # 目标网络的参数的更新频率
batch_size = 32
n_hidden = 128  # 隐含层神经元个数
min_size = 200  # 经验池超过200后再训练
return_list = []  # 记录每个回合的回报

# 加载环境
env = gym.make("CartPole-v1", render_mode="human")
n_states = env.observation_space.shape[0]  # 4
n_actions = env.action_space.n  # 2

# 实例化经验池
replay_buffer = ReplayBuffer(capacity)
# 实例化DQN
agent = DQN(n_states=n_states,
            n_hidden=n_hidden,
            n_actions=n_actions,
            learning_rate=lr,
            gamma=gamma,
            epsilon=epsilon,
            target_update=target_update,
            device=device,
        )

# 训练模型
for i in range(100):  # 100回合
    # 每个回合开始前重置环境
    state = env.reset()[0]  # len=4
    # 记录每个回合的回报
    episode_return = 0
    done = False
    
    # 打印训练进度，一共10回合
    with tqdm(total=10, desc='Iteration %d' % i) as pbar:

        while True:
            # 获取当前状态下需要采取的动作
            action = agent.take_action(state)
            # 更新环境
            next_state, reward, done, _, _ = env.step(action)
            # 添加经验池
            replay_buffer.add(state, action, reward, next_state, done)
            # 更新当前状态
            state = next_state
            # 更新回合回报
            episode_return += reward

            # 当经验池超过一定数量后，训练网络
            if replay_buffer.size() > min_size:
                # 从经验池中随机抽样作为训练集
                s, a, r, ns, d = replay_buffer.sample(batch_size)
                # 构造训练集
                transition_dict = {
                    'states': s,
                    'actions': a,
                    'next_states': ns,
                    'rewards': r,
                    'dones': d,
                }
                # 网络更新
                agent.update(transition_dict)
            # 找到目标就结束
            if done: break
        
        # 记录每个回合的回报
        return_list.append(episode_return)

        # 更新进度条信息
        pbar.set_postfix({
            'return': '%.3f' % return_list[-1]
        })
        pbar.update(1)

# 绘图
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN Returns')
plt.show()
