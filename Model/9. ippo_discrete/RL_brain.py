# 和PPO离散模型基本一致
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

# ----------------------------------------- #
# 策略网络--actor
# ----------------------------------------- #

class PolicyNet(nn.Module):  # 输入当前状态，输出动作的概率分布
    def __init__(self, n_states, n_hiddens, n_actions):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_hiddens)
        self.fc3 = nn.Linear(n_hiddens, n_actions)
    def forward(self, x):  # [b,n_states]
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x = F.relu(x)
        x = self.fc2(x)  # [b,n_hiddens]-->[b,n_hiddens]
        x = F.relu(x)
        x = self.fc3(x)  # [b,n_hiddens]-->[b,n_actions]
        x = F.softmax(x, dim=1)  # 每种动作选择的概率
        return x

# ----------------------------------------- #
# 价值网络--critic
# ----------------------------------------- #

class ValueNet(nn.Module):  # 评价当前状态的价值
    def __init__(self, n_states, n_hiddens):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_hiddens)
        self.fc3 = nn.Linear(n_hiddens, 1)
    def forward(self, x):  # [b,n_states]
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x = F.relu(x)
        x = self.fc2(x)  # [b,n_hiddens]-->[b,n_hiddens]
        x = F.relu(x)
        x = self.fc3(x)  # [b,n_hiddens]-->[b,1]
        return x

# ----------------------------------------- #
# 模型构建
# ----------------------------------------- #

class PPO:
    def __init__(self, n_states, n_hiddens, n_actions,
                 actor_lr, critic_lr, 
                 lmbda, eps, gamma, device):
        # 属性分配
        self.n_hiddens = n_hiddens
        self.actor_lr = actor_lr  # 策略网络的学习率
        self.critic_lr = critic_lr  # 价值网络的学习率
        self.lmbda = lmbda  # 优势函数的缩放因子
        self.eps = eps  # ppo截断范围缩放因子
        self.gamma = gamma  # 折扣因子
        self.device = device
        # 网络实例化
        self.actor = PolicyNet(n_states, n_hiddens, n_actions).to(device)  # 策略网络
        self.critic = ValueNet(n_states, n_hiddens).to(device)  # 价值网络
        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
    
    # 动作选择
    def take_action(self, state):  # [n_states]
        state = torch.tensor([state], dtype=torch.float).to(self.device)  # [1,n_states]
        probs = self.actor(state)  # 当前状态的动作概率 [b,n_actions]
        action_dist = torch.distributions.Categorical(probs)  # 构造概率分布
        action = action_dist.sample().item()  # 从概率分布中随机取样 int
        return action
    
    # 训练
    def update(self, transition_dict):
        # 取出数据集
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)  # [b,n_states]
        actions = torch.tensor(transition_dict['actions']).view(-1,1).to(self.device)  # [b,1]
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)  # [b,n_states]
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1,1).to(self.device)  # [b,1]
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1,1).to(self.device)  # [b,1]

        # 价值网络
        next_state_value = self.critic(next_states)  # 下一时刻的state_value  [b,1]
        td_target = rewards + self.gamma * next_state_value * (1-dones)  # 目标--当前时刻的state_value  [b,1]
        td_value = self.critic(states)  # 预测--当前时刻的state_value  [b,1]
        td_delta = td_value - td_target  # 时序差分  # [b,1]

        # 计算GAE优势函数，当前状态下某动作相对于平均的优势
        advantage = 0  # 累计一个序列上的优势函数
        advantage_list = []  # 存放每个时序的优势函数值
        td_delta = td_delta.cpu().detach().numpy()  # gpu-->numpy
        for delta in td_delta[::-1]:  # 逆序取出时序差分值
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append(advantage)  # 保存每个时刻的优势函数
        advantage_list.reverse()  # 正序
        advantage = torch.tensor(advantage_list, dtype=torch.float).to(self.device)

        # 计算当前策略下状态s的行为概率 / 在之前策略下状态s的行为概率
        old_log_probs = torch.log(self.actor(states).gather(1,actions))  # [b,1]
        log_probs = torch.log(self.actor(states).gather(1,actions))
        ratio = log_probs / old_log_probs

        # clip截断
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * advantage
        
        # 损失计算
        actor_loss = torch.mean(-torch.min(surr1, surr2))  # clip截断
        critic_loss = torch.mean(F.mse_loss(td_value, td_target))  # 
        # 梯度更新
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
