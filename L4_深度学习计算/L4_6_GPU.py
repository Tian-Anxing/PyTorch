import torch
from torch import nn

#查看GPU是否可用
print(torch.cuda.is_available()) # 输出 True

#查看GPU数量
print(torch.cuda.device_count()) # 输出 1

#查看当前GPU索引号，索引号从0开始
print(torch.cuda.current_device()) # 输出 0

#根据索引号查看GPU名字
print(torch.cuda.get_device_name(0))

#Tensor的GPU计算
x = torch.tensor([1, 2, 3])
print(x)
x = x.cuda(0) #或x = x.cuda()
print(x)
#查看数据位置
print(x.device)
#我们可以直接在创建的时候就指定设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.tensor([1, 2, 3], device=device)
# or
x = torch.tensor([1, 2, 3]).to(device)
print(x)
#如果在GPU上的数据进行运算，那么结果还是存放在GPU上的
y = x**2
print(y)

#需要注意的是，存储在不同位置中的数据是不可以直接进行计算的。
# z = y + x.cpu()#报错

#模型的GPU计算
net = nn.Linear(3, 1)
print(list(net.parameters())[0].device)
#将模型转到GPU上
#list(net.parameters())[0].cuda（）
net.cuda()
list(net.parameters())[0].device
#同样的我们需要保证模型输入的Tensor也在GPU上避免报错
x = torch.rand(2,3).cuda()
a = net(x)
print(a)