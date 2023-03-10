import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import collections
import random

# --------------------------------------- #
# 经验回放池
# --------------------------------------- #

class ReplayBuffer():
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
    # 将数据以元组形式添加进经验池
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    # 随机采样batch_size行数据
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)  # list, len=32
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done
    # 目前队列长度
    def size(self):
        return len(self.buffer)
   
# -------------------------------------- #
# 构造深度学习网络，输入状态s，得到各个动作的reward
# -------------------------------------- #

class Net(nn.Module):
    def __init__(self, n_states, n_hidden, n_actions):
        super(Net, self).__init__()
        # [b,n_states]-->[b,n_hidden]
        self.fc1 = nn.Linear(n_states, n_hidden)
        # [b,n_hidden]-->[b,n_actions]
        self.fc2 = nn.Linear(n_hidden, n_actions)
    def forward(self, x):  # [b,n_states]
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# -------------------------------------- #
# 构造深度强化学习模型
# -------------------------------------- #

class DQN:
    def __init__(self, n_states, n_hidden, n_actions,
                 learning_rate, gamma, epsilon,
                 target_update, device):
        self.n_states = n_states  # 状态的特征数
        self.n_hidden = n_hidden  # 隐含层个数
        self.n_actions = n_actions  # 动作数
        self.learning_rate = learning_rate  # 训练时的学习率
        self.gamma = gamma  # 折扣因子，对下一状态的回报的缩放
        self.epsilon = epsilon  # 贪婪策略，有1-epsilon的概率探索
        self.target_update = target_update  # 目标网络的参数的更新频率
        self.device = device
        self.count = 0

        # 构建2个神经网络，相同的结构，不同的参数
        self.q_net = Net(self.n_states, self.n_hidden, self.n_actions)  # 实例化训练网络
        self.target_q_net = Net(self.n_states, self.n_hidden, self.n_actions)  # 实例化目标网络
        # 优化器，更新训练网络的参数
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate)

    #（2）动作选择
    def take_action(self, state):
        state = torch.Tensor(state[np.newaxis, :])
        if np.random.random() < self.epsilon:  # 0-1
            actions_value = self.q_net(state)
            action = actions_value.argmax().item()  # int
        else:
            action = np.random.randint(self.n_actions)
        return action

    #（3）网络训练
    def update(self, transition_dict):  # 传入经验池中的batch个样本
        states = torch.tensor(transition_dict['states'], dtype=torch.float)
        actions = torch.tensor(transition_dict['actions']).view(-1,1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1,1)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1,1)

        q_values = self.q_net(states).gather(1, actions)  # [b,1]
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1,1)
        q_targets = rewards + self.gamma * max_next_q_values * (1-dones)

        # 目标网络和训练网络之间的均方误差损失
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        # 在一段时间后更新目标网络的参数
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())
        
        self.count += 1
