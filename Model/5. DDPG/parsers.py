# 参数定义
import argparse  # 参数设置

# 创建解释器
parser = argparse.ArgumentParser()

# 参数定义
parser.add_argument('--actor_lr', type=float, default=3e-4, help='策略网络的学习率')
parser.add_argument('--critic_lr', type=float, default=3e-3, help='价值网络的学习率')
parser.add_argument('--n_hiddens', type=int, default=64, help='隐含层神经元个数')
parser.add_argument('--gamma', type=float, default=0.98, help='折扣因子')
parser.add_argument('--tau', type=float, default=0.005, help='软更新系数')
parser.add_argument('--buffer_size', type=int, default=1000, help='经验池容量')
parser.add_argument('--min_size', type=int, default=200, help='经验池超过200再训练')
parser.add_argument('--batch_size', type=int, default=64, help='每次训练64组样本')
parser.add_argument('--sigma', type=int, default=0.01, help='高斯噪声标准差')

# 参数解析
args=parser.parse_args()
