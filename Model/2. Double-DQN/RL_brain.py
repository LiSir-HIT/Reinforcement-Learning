# 基于策略的学习方法，用于数值连续的问题
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# ----------------------------------------------------- #
#（1）构建训练网络
# ----------------------------------------------------- #
class Net(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions):
        super(Net, self).__init__()
        # 只有一层隐含层的网络
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_actions)
    # 前向传播
    def forward(self, x):
        x = self.fc1(x)  # [b, states]==>[b, n_hiddens]
        x = F.relu(x)
        x = self.fc2(x)  # [b, n_hiddens]==>[b, n_actions]
        # 对batch中的每一行样本计算softmax，q值越大，概率越大
        x = F.softmax(x, dim=1)  # [b, n_actions]==>[b, n_actions]
        return x

# ----------------------------------------------------- #
#（2）强化学习模型
# ----------------------------------------------------- #
class PolicyGradient:
    def __init__(self, n_states, n_hiddens, n_actions, 
                 learning_rate, gamma):
        # 属性分配
        self.n_states = n_states  # 状态数
        self.n_hiddens = n_hiddens
        self.n_actions = n_actions  # 动作数
        self.learning_rate = learning_rate  # 衰减
        self.gamma = gamma  # 折扣因子
        self._build_net()  # 构建网络模型

    # 网络构建
    def _build_net(self):
        # 网络实例化
        self.policy_net = Net(self.n_states, self.n_hiddens, self.n_actions)
        # 优化器
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

    # 动作选择，根据概率分布随机采样
    def take_action(self, state):  # 传入某个人的状态
        # numpy[n_states]-->[1,n_states]-->tensor
        state = torch.Tensor(state[np.newaxis, :])
        # 获取每个人的各动作对应的概率[1,n_states]-->[1,n_actions]
        probs = self.policy_net(state)
        # 创建以probs为标准类型的数据分布
        action_dist = torch.distributions.Categorical(probs)
        # 以该概率分布随机抽样 [1,n_actions]-->[1] 每个状态取一组动作
        action = action_dist.sample()
        # 将tensor数据变成一个数 int
        action = action.item()
        return action

    # 获取每个状态最大的state_value
    def max_q_value(self, state):
        # 维度变换[n_states]-->[1,n_states]
        state = torch.tensor(state, dtype=torch.float).view(1,-1)
        # 获取状态对应的每个动作的reward的最大值 [1,n_states]-->[1,n_actions]-->[1]-->float
        max_q = self.policy_net(state).max().item()
        return max_q

    # 训练模型
    def learn(self, transitions_dict):  # 输入batch组状态[b,n_states]
        # 取出该回合中所有的链信息
        state_list = transitions_dict['states']
        action_list = transitions_dict['actions']
        reward_list = transitions_dict['rewards']

        G = 0  # 记录该条链的return
        self.optimizer.zero_grad()  # 优化器清0
        # 梯度上升最大化目标函数
        for i in reversed(range(len(reward_list))):
            # 获取每一步的reward, float
            reward = reward_list[i]
            # 获取每一步的状态 [n_states]-->[1,n_states]
            state = torch.tensor(state_list[i], dtype=torch.float).view(1,-1)
            # 获取每一步的动作 [1]-->[1,1]
            action = torch.tensor(action_list[i]).view(1,-1)
            # 当前状态下的各个动作价值函数 [1,2]
            q_value = self.policy_net(state)
            # 获取已action对应的概率 [1,1]
            log_prob = torch.log(q_value.gather(1, action))
            # 计算当前状态的state_value = 及时奖励 + 下一时刻的state_value
            G = reward + self.gamma * G
            # 计算每一步的损失函数
            loss = -log_prob * G
            # 反向传播
            loss.backward()
        # 梯度下降
        self.optimizer.step()
