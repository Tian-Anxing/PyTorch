import torch
from torch import nn
from d2lzh_pytorch import corr2d
import time

#二维卷积层  卷积核或过滤器,卷积核窗口（又称卷积窗口）的形状取决于卷积核的高和宽
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

X = torch.ones(6, 8)
X[:, 2:6] = 0
# #然后我们构造一个高和宽分别为1和2的卷积核K。当它与输入做互相关运算时，如果横向相邻元素相同，输出为0；否则输出为非0。
K = torch.tensor([[1, -1]])
Y = corr2d(X, K)
# print(Y)

conv2d = Conv2D(kernel_size=(1, 2))

step = 40
lr = 0.01
for i in range(step):
    Y_hat = conv2d(X)
    l = ((Y_hat - Y) ** 2).sum()
    l.backward()

    # 梯度下降
    conv2d.weight.data -= lr * conv2d.weight.grad
    conv2d.bias.data -= lr * conv2d.bias.grad

    # 梯度清0
    conv2d.weight.grad.fill_(0)
    conv2d.bias.grad.fill_(0)
    if (i + 1) % 5 == 0:
        print('Step %d, loss %.3f' % (i + 1, l.item()))
print("weight: ", conv2d.weight.data)
print("bias: ", conv2d.bias.data)