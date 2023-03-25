#根据自变量在每个维度的梯度值的大小来调整各个维度上的学习率，从而避免统一的学习率难以适应所有维度的问题
#这里开方、除法和乘法的运算都是按元素运算的。这些按元素运算使得目标函数自变量中每个元素都分别拥有自己的学习率。
#使用AdaGrad算法时，自变量中每个元素的学习率在迭代过程中一直在降低（或不变）
import math
import torch
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

def adagrad_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6  # 前两项为自变量梯度
    s1 += g1 ** 2
    s2 += g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta = 0.4
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))

eta = 2
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))

#简介实现  通过名称为Adagrad的优化器方法，我们便可使用PyTorch提供的AdaGrad算法来训练模型。
features, labels = d2l.get_data_ch7()
d2l.train_pytorch_ch7(torch.optim.Adagrad, {'lr': 0.1}, features, labels)