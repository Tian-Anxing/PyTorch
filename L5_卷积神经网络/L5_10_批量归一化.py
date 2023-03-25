#我们对输入数据做了标准化处理：处理后的任意一个特征在数据集中所有样本上的均值为0、标准差为1。标准化处理输入数据使各个特征的分布相近
#在模型训练时，批量归一化利用小批量上的均值和标准差，不断调整神经网络中间输出，从而使整个神经网络在各层的中间输出的数值更稳定。

#对全连接层做批量归一化
#通常，我们将批量归一化层置于全连接层中的仿射变换和激活函数之间。批量归一化层的输出同样是d维向量
#并由以下几步求得。首先，对小批量B求均值和方差：其中的平方计算是按元素求平方。接下来，使用按元素开方和按元素除法对x(i)标准化：
#批量归一化层引入了两个可以学习的模型参数，拉伸（scale）参数 γ和偏移（shift）参数 β。
#这两个参数和x(i)形状相同，皆为d维向量。它们与x(i)分别做按元素乘法（符号⊙）和加法计算

#值得注意的是，可学习的拉伸和偏移参数保留了不对xˆ(i)做批量归一化的可能，如果批量归一化无益，理论上，学出的模型可以不使用批量归一化。   为啥啊



#对卷积层做批量归一化
#对卷积层来说，批量归一化发生在卷积计算之后、应用激活函数之前。如果卷积计算输出多个通道，我们需要对这些通道的输出分别做批量归一化，
#且每个通道都拥有独立的拉伸和偏移参数，并均为标量。设小批量中有m个样本。在单个通道上，假设卷积计算输出的高和宽分别为p和q。
#我们需要对该通道中m×p×q个元素同时做批量归一化。对这些元素做标准化计算时，我们使用相同的均值和方差，即该通道中m×p×q个元素的均值和方差。


#预测时的批量归一化
#使用批量归一化训练时，我们可以将批量大小设得大一点，从而使批量内样本的均值和方差的计算都较为准确。将训练好的模型用于预测时，我们希望模型对于任意输入都有确定的输出。
#因此，单个样本的输出不应取决于批量归一化所需要的随机小批量中的均值和方差。一种常用的方法是通过移动平均估算整个训练数据集的样本均值和方差，并在预测时使用它们得到确定的输出。
#可见，和丢弃层一样，批量归一化层在训练模式和预测模式下的计算结果也是不一样的。


#从零实现
import time
import torch
from torch import nn, optim
import torch.nn.functional as F
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#批量归一化层
def batch_norm(is_training, X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 判断当前模式是训练模式还是预测模式
    if not is_training:
        # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使用全连接层的情况，计算特征维上的均值和方差
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。这里我们需要保持
            # X的形状以便后面可以做广播运算
            mean = X.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            var = ((X - mean) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        # 训练模式下用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)#eps为防止sqrt内数字为0
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # 拉伸和偏移
    return Y, moving_mean, moving_var

#定义一个BatchNorm层
class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super(BatchNorm, self).__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成0和1
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 不参与求梯度和迭代的变量，全在内存上初始化成0
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)

    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var, Module实例的traning属性默认为true, 调用.eval()后设成false
        Y, self.moving_mean, self.moving_var = batch_norm(self.training,
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y

#使用数字归一化的LeNet层
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.Lenet = nn.Sequential(
            nn.Conv2d(1, 6, 5),  # in_channels, out_channels, kernel_size
            BatchNorm(6,4),
            nn.Sigmoid(),
            nn.MaxPool2d((2, 2), stride=2),  # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            BatchNorm(16, 4),
            nn.Sigmoid(),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120),  # 为什么4*4,为什么扩到16
            BatchNorm(120, 2),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            BatchNorm(84, 2),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self,mig):
        return self.Lenet(mig)

#或
# net = nn.Sequential(
#             nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
#             BatchNorm(6, num_dims=4),
#             nn.Sigmoid(),
#             nn.MaxPool2d(2, 2), # kernel_size, stride
#             nn.Conv2d(6, 16, 5),
#             BatchNorm(16, num_dims=4),
#             nn.Sigmoid(),
#             nn.MaxPool2d(2, 2),
#             nn.Flatten(),
#             nn.Linear(16*4*4, 120),
#             BatchNorm(120, num_dims=2),
#             nn.Sigmoid(),
#             nn.Linear(120, 84),
#             BatchNorm(84, num_dims=2),
#             nn.Sigmoid(),
#             nn.Linear(84, 10)
#         )

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
net = LeNet()
lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

#查看批量归一化层学习到的拉伸参数gamma和偏移参数beta
print(list(net.parameters())[1] ).gamma.view((-1,), list(net.parameters())[1] ).beta.view((-1,))

#简洁实现
# net = nn.Sequential(
#             nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
#             nn.BatchNorm2d(6),
#             nn.Sigmoid(),
#             nn.MaxPool2d(2, 2), # kernel_size, stride
#             nn.Conv2d(6, 16, 5),
#             nn.BatchNorm2d(16),
#             nn.Sigmoid(),
#             nn.MaxPool2d(2, 2),
#             nn.Flatten(),
#             nn.Linear(16*4*4, 120),
#             nn.BatchNorm1d(120),
#             nn.Sigmoid(),
#             nn.Linear(120, 84),
#             nn.BatchNorm1d(84),
#             nn.Sigmoid(),
#             nn.Linear(84, 10)
#         )
#
# batch_size = 256
# train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
#
# lr, num_epochs = 0.001, 5
# optimizer = torch.optim.Adam(net.parameters(), lr=lr)
# d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)