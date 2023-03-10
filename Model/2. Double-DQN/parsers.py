# 参数定义
import argparse  # 参数设置

# 创建解释器
parser = argparse.ArgumentParser()

# 参数定义
parser.add_argument('--lr', type=float, default=2e-3, help='学习率')
parser.add_argument('--gamma', type=float, default=0.9, help='折扣因子')
parser.add_argument('--epsilon', type=float, default=0.9, help='贪心系数')
parser.add_argument('--target_update', type=int, default=200, help='更新频率')
parser.add_argument('--batch_size', type=int, default=64, help='每次训练64组数据')
parser.add_argument('--capacity', type=int, default=500, help='经验池容量')
parser.add_argument('--min_size', type=int, default=200, help='经验池超过200后再开始训练')
parser.add_argument('--n_hiddens', type=int, default=128, help='隐含层神经元个数')

# 参数解析
args=parser.parse_args()
