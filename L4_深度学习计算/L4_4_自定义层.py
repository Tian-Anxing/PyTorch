import torch
from torch import nn

#不含模型参数的自定义层
# class CenteredLayer(nn.Module):
#     def __init__(self, **kwargs):
#         super(CenteredLayer, self).__init__(**kwargs)
#     def forward(self, x):
#         return x - x.mean()
# layer = CenteredLayer()
# layer(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float))
# # print(a)
# net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
# y = net(torch.rand(4, 8))
# a = int(y.mean().item())
# print(type(a))
# print(a)

#含模型参数的自定义层
#ParameterList接收一个Parameter实例的列表作为输入然后得到一个参数列表，使用的时候可以用索引来访问某个参数，另外也可以使用append和extend在列表后面新增参数。
class MyListDense(nn.Module):
    def __init__(self):
        super(MyListDense, self).__init__()
        self.params = nn.ParameterList([nn.Parameter(torch.randn(4, 4)) for i in range(3)])
        self.params.append(nn.Parameter(torch.randn(4, 1)))

    def forward(self, x):
        for i in range(len(self.params)):
            x = torch.mm(x, self.params[i])
        return x
# net = MyDense()
# print(net)

#ParameterDict接收一个Parameter实例的字典作为输入然后得到一个参数字典，然后可以按照字典的规则使用了。例如使用update()新增参数，使用keys()返回所有键值，使用items()返回所有键值对等等
class MyDictDense(nn.Module):
    def __init__(self):
        super(MyDictDense, self).__init__()
        self.params = nn.ParameterDict({
                'linear1': nn.Parameter(torch.randn(4, 4)),
                'linear2': nn.Parameter(torch.randn(4, 1))
        })
        self.params.update({'linear3': nn.Parameter(torch.randn(4, 2))}) # 新增

    def forward(self, x, choice='linear1'):#当未传入层数参数时使用第一层
        return torch.mm(x, self.params[choice])

net = MyDictDense()
print(net)

#这样就可以根据传入的键值来进行不同的前向传播：
x = torch.ones(1, 4)
# print(net(x, 'linear1'))
# print(net(x, 'linear2'))
# print(net(x, 'linear3'))

#我们也可以使用自定义层构造模型。它和PyTorch的其他层在使用上很类似。
net = nn.Sequential(
    MyDictDense(),
    MyListDense(),
)
print(net)
print(net(x))