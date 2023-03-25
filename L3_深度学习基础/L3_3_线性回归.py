import torch
import numpy as np
import torch.utils.data as Data
from torch import nn
from LinearNet import *
from torch.nn import init
import torch.optim as optim

#生成数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

#读取数据集
batch_size = 10
dataset = Data.TensorDataset(features,labels)
# print(features)
# print(labels)
# print(dataset)
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
# for X, y in data_iter:
#     print(X, y)
#     break
# net = LinearNet(num_inputs)

#定义模型
net = nn.Sequential(
    nn.Linear(num_inputs, 1)
    # ,nn.Linear(2,1)  #两个参数一个是输入特征量 一个是输出特征量
    # 此处还可以传入其他层
    )
# print(net)  # 使用print可以打印出网络的结构
# for param in net.parameters():
#     print(param)

#初始化模型参数
init.normal_(net[0].weight, mean=0, std=0.01)
init.constant_(net[0].bias, val=0)  # 也可以直接修改bias的data: net[0].bias.data.fill_(0)
# print(net[0].weight)
# print(net[0].bias)

#定义损失函数
loss = nn.MSELoss()

#定义优化算法
optimizer = optim.SGD(net.parameters(), lr=0.03)
# optimizer =optim.SGD([
#                 # 如果对某个参数不指定学习率，就使用最外层的默认学习率
#                 {'params': net.subnet1.parameters()}, # lr=0.03
#                 {'params': net.subnet2.parameters(), 'lr': 0.01}
#             ], lr=0.03)
# # 调整学习率
# for param_group in optimizer.param_groups:
#     param_group['lr'] *= 0.1 # 学习率为之前的0.1倍
# print(optimizer)

#训练模型
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))
dense = net[0]
print(true_w,dense.weight)
print(net[0].bias)

#torch.utils.data模块提供了有关数据处理的工具，torch.nn模块定义了大量神经网络的层，
#torch.nn.init模块定义了各种初始化方法，torch.optim模块提供了很多常用的优化算法。