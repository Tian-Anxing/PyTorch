from torch import nn
from collections import OrderedDict

class LinearNet(nn.Module):#nn.Module为继承
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()#用于调用父类(超类)的一个方法。
        self.linear = nn.Linear(num_inputs, num_outputs)
        # forward 定义前向传播
    def forward(self, x): # x shape: (batch, 1, 28, 28)
        y = self.linear(x.view(x.shape[0], -1))
        return y

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)