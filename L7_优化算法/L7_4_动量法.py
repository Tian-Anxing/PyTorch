import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
import torch

#设置一个超参数 在改变参数值时不仅受梯度和学习率的影响，还受该超参数和前一时刻的改变量影响
eta = 0.4 # 学习率
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)
d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
eta = 0.6
d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))

#动量法
def momentum_2d(x1, x2, v1, v2):
    v1 = gamma * v1 + eta * 0.2 * x1
    v2 = gamma * v2 + eta * 4 * x2
    return x1 - v1, x2 - v2, v1, v2
eta, gamma = 0.4, 0.5
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
eta = 0.6
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
#在PyTorch中只需要通过参数momentum来指定动量超参数即可使用动量法。
features, labels = d2l.get_data_ch7()
d2l.train_pytorch_ch7(torch.optim.SGD, {'lr': 0.004, 'momentum': 0.9},
                    features, labels)