import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
sys.path.append("../..")
import d2lzh_pytorch as d2l
import LinearNet
from collections import OrderedDict

#获取和读取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

#定义和初始化模型
num_inputs = 784
num_outputs = 10

net = LinearNet.LinearNet(num_inputs, num_outputs)

net = nn.Sequential(#有其他写法 详见原网页3.3
    # FlattenLayer(),
    # nn.Linear(num_inputs, num_outputs)
    OrderedDict([
        ('flatten', LinearNet.FlattenLayer()),
        ('linear', nn.Linear(num_inputs, num_outputs))
    ])
)

init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

#softmax和交叉熵损失函数
loss = nn.CrossEntropyLoss()

#定义优化算法
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

#训练模型
num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)