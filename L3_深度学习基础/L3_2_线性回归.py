#import D2_3_Tensor
import torch
from time import time
import sys
import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
from d2lzh_pytorch import *

# a = torch.ones(1000000)
# b = torch.ones(1000000)
# start = time()
# c = a+b
# end = time()
# print(end-start)
#超参数：批量大小和学习率的值是人为设定的，并不是通过模型训练出来的，因此叫做超参数。

num_inputs = 2
num_examples = 1000
true_w = [2,-3.4]
true_b = 4.2
features = torch.randn(num_examples, num_inputs,
                       dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                       dtype=torch.float32)
# set_figsize()
# plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
# plt.show()
# #生成数据

# torch.manual_seed(100)
# x = torch.randn(2,3,4)
# print(x)
# b = torch.index_select(x,1,torch.LongTensor([0]))
# print('bbbbbb',b)


# def foo():
#     print("starting...")
#     while True:
#         res = yield 4
#         print("res:",res)
# g = foo()
# print(next(g))
# print("*"*20)
# print(next(g))

batch_size = 10
# for X,y in data_iter(batch_size,features,labels):
#     print(X,y)
#     break

lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

#
w = torch.tensor([[2],[2]],dtype=torch.float32)
b = torch.tensor(1,dtype=torch.float32)
# w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
print(w)
# b = torch.zeros(1, dtype=torch.float32)
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)
for epoch in range(num_epochs):  # 训练模型一共需要num_epochs个迭代周期
    # 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）。X
    # 和y分别是小批量样本的特征和标签
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()  # l是有关小批量X和y的损失
        l.backward()  # 小批量的损失对模型参数求梯度
        sgd([w, b], lr, batch_size)  # 使用小批量随机梯度下降迭代模型参数

        # 不要忘了梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))

print(true_w, '\n', w)
print(true_b, '\n', b)