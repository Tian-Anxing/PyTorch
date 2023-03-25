import torch
from torch import nn
from torch.nn import init


#
net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))  # pytorch已进行默认初始化

# print(net)
# X = torch.rand(2, 4)
# Y = net(X).sum()


#访问模型参数
# print(type(net.named_parameters()))
# for name, param in net.named_parameters():
#     print(name, param.size())
#
# for name, param in net[0].named_parameters():
#     print(name, param.size(), type(param))

class MyModel(nn.Module):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.weight1 = nn.Parameter(torch.rand(20, 20))
        self.weight2 = torch.rand(20, 20)
    def forward(self, x):
        pass
#
# n = MyModel()
# for name, param in n.named_parameters():
#     print(name)

#另外返回的param的类型为torch.nn.parameter.Parameter，其实这是Tensor的子类，和Tensor不同的是如果一个Tensor是Parameter，那么它会自动被添加到模型的参数列表里
class MyModel(nn.Module):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.weight1 = nn.Parameter(torch.rand(20, 20))
        self.weight2 = torch.rand(20, 20)
    def forward(self, x):
        pass

# n = MyModel()
# for name, param in n.named_parameters():
#     print(name)

# weight_0 = list(net[0].parameters())[0]
# print(weight_0.data)
# print(weight_0.grad) # 反向传播前梯度为None
# Y.backward()
# print(weight_0.grad)

#初始化模型参数
# for name, param in net.named_parameters():
#     if 'weight' in name:
#         init.normal_(param, mean=0, std=0.01)
#         print(name, param.data)
#
# for name, param in net.named_parameters():
#     if 'bias' in name:
#         init.constant_(param, val=0)
#         print(name, param.data)

#自定义初始化方法
# def normal_(tensor, mean=0, std=1):
#     with torch.no_grad():#不跟踪返回梯度
#         return tensor.normal_(mean, std)
#
# def init_weight_(tensor):
#     with torch.no_grad():
#         tensor.uniform_(-10,10)
#         tensor *= (tensor.abs() >= 5).float()
#
# for name, param in net.named_parameters():
#     if 'weight' in name:
#         init_weight_(param)
#         print(name, param.data)
# for name, param in net.named_parameters():
#     if 'bias' in name:
#         param.data += 1#通过对修改参数的data来来写模型参数同时不会影响梯度
#         print(name, param.data)
# i = 0
# for name,param in net.named_parameters():
#     print(name)
#     print(i)
#     i+=1


# 共享模型参数
linear = nn.Linear(1, 1, bias=False)
net = nn.Sequential(linear, linear)
print(net)
for name, param in net.named_parameters():
    init.constant_(param, val=3)
    print(name, param.data)
print(id(net[0]) == id(net[1]))
print(id(net[0].weight) == id(net[1].weight))
#因为模型参数里包含了梯度，所以在反向传播计算时，这些共享的参数的梯度是累加的:
x = torch.ones(1, 1)
y = net(x).sum()
print(y)
y.backward()
print(net[0].weight.grad) # 单次梯度是3，两次所以就是6  xian'du