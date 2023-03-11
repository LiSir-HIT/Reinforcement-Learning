import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import collections
import random

# ------------------------------------- #
# 经验回放池
# ------------------------------------- #

class ReplayBuffer:
    def __init__(self, capacity):  # 经验池的最大容量
        # 创建一个队列，先进先出
        self.buffer = collections.deque(maxlen=capacity)
    # 在队列中添加数据
    def add(self, state, action, reward, next_state, done):
        # 以list类型保存
        self.buffer.append((state, action, reward, next_state, done))
    # 在队列中随机取样batch_size组数据
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        # 将数据集拆分开来
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done
    # 测量当前时刻的队列长度
    def size(self):
        return len(self.buffer)

# ------------------------------------- #
# 策略网络
# ------------------------------------- #

class PolicyNet(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions, action_bound):
        super(PolicyNet, self).__init__()
        # 环境可以接受的动作最大值
        self.action_bound = action_bound
        # 只包含一个隐含层
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_actions)
    # 前向传播
    def forward(self, x):
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x = F.relu(x)
        x = self.fc2(x)  # [b,n_hiddens]-->[b,n_actions]
        x= torch.tanh(x)  # 将数值调整到 [-1,1]
        x = x * self.action_bound  # 缩放到 [-action_bound, action_bound]
        return x

# ------------------------------------- #
# 价值网络
# ------------------------------------- #

class QValueNet(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions):
        super(QValueNet, self).__init__()
        # 
        self.fc1 = nn.Linear(n_states + n_actions, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_hiddens)
        self.fc3 = nn.Linear(n_hiddens, 1)
    # 前向传播
    def forward(self, x, a):
        # 拼接状态和动作
        cat = torch.cat([x, a], dim=1)  # [b, n_states + n_actions]
        x = self.fc1(cat)  # -->[b, n_hiddens]
        x = F.relu(x)
        x = self.fc2(x)  # -->[b, n_hiddens]
        x = F.relu(x)
        x = self.fc3(x)  # -->[b, 1]
        return x

# ------------------------------------- #
# 算法主体
# ------------------------------------- #

class DDPG:
    def __init__(self, n_states, n_hiddens, n_actions, action_bound,
                 sigma, actor_lr, critic_lr, tau, gamma, device):

        # 策略网络--训练
        self.actor = PolicyNet(n_states, n_hiddens, n_actions, action_bound).to(device)
        # 价值网络--训练
        self.critic = QValueNet(n_states, n_hiddens, n_actions).to(device)
        # 策略网络--目标
        self.target_actor = PolicyNet(n_states, n_hiddens, n_actions, action_bound).to(device)
        # 价值网络--目标
        self.target_critic = QValueNet(n_states, n_hiddens, n_actions).to(device
                                                                          )
        # 初始化价值网络的参数，两个价值网络的参数相同
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 初始化策略网络的参数，两个策略网络的参数相同
        self.target_actor.load_state_dict(self.actor.state_dict())

        # 策略网络的优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        # 价值网络的优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 属性分配
        self.gamma = gamma  # 折扣因子
        self.sigma = sigma  # 高斯噪声的标准差，均值设为0
        self.tau = tau  # 目标网络的软更新参数
        self.n_actions = n_actions
        self.device = device

    # 动作选择
    def take_action(self, state):
        # 维度变换 list[n_states]-->tensor[1,n_states]-->gpu
        state = torch.tensor(state, dtype=torch.float).view(1,-1).to(self.device)
        # 策略网络计算出当前状态下的动作价值 [1,n_states]-->[1,1]-->int
        action = self.actor(state).item()
        # 给动作添加噪声，增加搜索
        action = action + self.sigma * np.random.randn(self.n_actions)
        return action
    
    # 软更新, 意思是每次learn的时候更新部分参数
    def soft_update(self, net, target_net):
        # 获取训练网络和目标网络需要更新的参数
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            # 训练网络的参数更新要综合考虑目标网络和训练网络
            param_target.data.copy_(param_target.data*(1-self.tau) + param.data*self.tau)

    # 训练
    def update(self, transition_dict):
        # 从训练集中取出数据
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)  # [b,n_states]
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1,1).to(self.device)  # [b,1]
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1,1).to(self.device)  # [b,1]
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)  # [b,next_states]
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1,1).to(self.device)  # [b,1]
        
        # 价值目标网络获取下一时刻的每个动作价值[b,n_states]-->[b,n_actors]
        next_q_values = self.target_actor(next_states)
        # 策略目标网络获取下一时刻状态选出的动作价值 [b,n_states+n_actions]-->[b,1]
        next_q_values = self.target_critic(next_states, next_q_values)
        # 当前时刻的动作价值的目标值 [b,1]
        q_targets = rewards + self.gamma * next_q_values * (1-dones)
        
        # 当前时刻动作价值的预测值 [b,n_states+n_actions]-->[b,1]
        q_values = self.critic(states, actions)

        # 预测值和目标值之间的均方差损失
        critic_loss = torch.mean(F.mse_loss(q_values, q_targets))
        # 价值网络梯度
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 当前状态的每个动作的价值 [b, n_actions]
        actor_q_values = self.actor(states)
        # 当前状态选出的动作价值 [b,1]
        score = self.critic(states, actor_q_values)
        # 计算损失
        actor_loss = -torch.mean(score)
        # 策略网络梯度
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新策略网络的参数  
        self.soft_update(self.actor, self.target_actor)
        # 软更新价值网络的参数
        self.soft_update(self.critic, self.target_critic)
