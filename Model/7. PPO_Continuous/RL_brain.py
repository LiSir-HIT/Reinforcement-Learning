# 用于连续动作的PPO
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# ------------------------------------- #
# 策略网络--输出连续动作的高斯分布的均值和标准差
# ------------------------------------- #

class PolicyNet(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc_mu = nn.Linear(n_hiddens, n_actions)
        self.fc_std = nn.Linear(n_hiddens, n_actions)
    # 前向传播
    def forward(self, x):  # 
        x = self.fc1(x)  # [b, n_states] --> [b, n_hiddens]
        x = F.relu(x)
        mu = self.fc_mu(x)  # [b, n_hiddens] --> [b, n_actions]
        mu = 2 * torch.tanh(mu)  # 值域 [-2,2]
        std = self.fc_std(x)  # [b, n_hiddens] --> [b, n_actions]
        std = F.softplus(std)  # 值域 小于0的部分逼近0，大于0的部分几乎不变
        return mu, std

# ------------------------------------- #
# 价值网络 -- 评估当前状态的价值
# ------------------------------------- #

class ValueNet(nn.Module):
    def __init__(self, n_states, n_hiddens):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, 1)
    # 前向传播
    def forward(self, x):  
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x = F.relu(x)
        x = self.fc2(x)  # [b,n_hiddens]-->[b,1]
        return x

# ------------------------------------- #
# 模型构建--处理连续动作
# ------------------------------------- #

class PPO:
    def __init__(self, n_states, n_hiddens, n_actions,
                 actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        # 实例化策略网络
        self.actor = PolicyNet(n_states, n_hiddens, n_actions).to(device)
        # 实例化价值网络
        self.critic = ValueNet(n_states, n_hiddens).to(device)
        # 策略网络的优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        # 价值网络的优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # 属性分配
        self.lmbda = lmbda  # GAE优势函数的缩放因子
        self.epochs = epochs  # 一条序列的数据用来训练多少轮
        self.eps = eps  # 截断范围
        self.gamma = gamma  # 折扣系数
        self.device = device 
    
    # 动作选择
    def take_action(self, state):  # 输入当前时刻的状态
        # [n_states]-->[1,n_states]-->tensor
        state = torch.tensor(state[np.newaxis, :]).to(self.device)
        # 预测当前状态的动作，输出动作概率的高斯分布
        mu, std = self.actor(state)
        # 构造高斯分布
        action_dict = torch.distributions.Normal(mu, std)
        # 随机选择动作
        action = action_dict.sample().item()
        return [action]  # 返回动作值

    # 训练
    def update(self, transition_dict):
        # 提取数据集
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)  # [b,n_states]
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1,1).to(self.device)  # [b,1]
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1,1).to(self.device)  # [b,1]
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)  # [b,n_states]
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1,1).to(self.device)  # [b,1]
        
        # 价值网络--目标，获取下一时刻的state_value  [b,n_states]-->[b,1]
        next_states_target = self.critic(next_states)
        # 价值网络--目标，当前时刻的state_value  [b,1]
        td_target = rewards + self.gamma * next_states_target * (1-dones)
        # 价值网络--预测，当前时刻的state_value  [b,n_states]-->[b,1]
        td_value = self.critic(states)
        # 时序差分，预测值-目标值  # [b,1]
        td_delta = td_value - td_target

        # 对时序差分结果计算GAE优势函数
        td_delta = td_delta.cpu().detach().numpy()  # [b,1]
        advantage_list = []  # 保存每个时刻的优势函数
        advantage = 0  # 优势函数初始值
        # 逆序遍历时序差分结果，把最后一时刻的放前面
        for delta in td_delta[::-1]:
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append(advantage)
        # 正序排列优势函数
        advantage_list.reverse()
        # numpy --> tensor
        advantage = torch.tensor(advantage_list, dtype=torch.float).to(self.device)

        # 策略网络--预测，当前状态选择的动作的高斯分布
        mu, std = self.actor(states)  # [b,1]
        # 基于均值和标准差构造正态分布
        action_dists = torch.distributions.Normal(mu.detch(), std.detch())
        # 从正态分布中选择动作，并使用log函数
        old_log_prob = action_dists.log_prob(actions)

        # 一个序列训练epochs次
        for _ in range(self.epochs):
            # 预测当前状态下的动作
            mu, std = self.actor(states)
            # 构造正态分布
            action_dists = torch.distributions.Normal(mu, std)
            # 当前策略在 t 时刻智能体处于状态 s 所采取的行为概率
            log_prob = action_dists.log_prob(actions)
            # 计算概率的比值来控制新策略更新幅度
            ratio = torch.exp(log_prob - old_log_prob)
            
            # 公式的左侧项
            surr1 = ratio * advantage
            # 公式的右侧项，截断
            surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps)

            # 策略网络的损失PPO-clip
            actor_loss = torch.mean(-torch.min(surr1,surr2))
            # 价值网络的当前时刻预测值，与目标价值网络当前时刻的state_value之差
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detch()))

            # 优化器清0
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            # 梯度反传
            actor_loss.backward()
            critic_loss.backward()
            # 参数更新
            self.actor_optimizer.step()
            self.critic_optimizer.step()
