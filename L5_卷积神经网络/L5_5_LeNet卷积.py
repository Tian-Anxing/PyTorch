import time
import torch
from torch import nn, optim
import d2lzh_pytorch as d2l
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,6,5),# in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d((2,2),stride=2), # kernel_size, stride
            nn.Conv2d(6,16,5),
            nn.Sigmoid(),
            nn.MaxPool2d((2,2),stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4,120),#为什么4*4,为什么扩到16
            nn.Sigmoid(),
            nn.Linear(120,84),
            nn.Sigmoid(),
            nn.Linear(84,10)
        )
    def forward(self,img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0],-1))#在单通道时img.shapr[0]为批量大小，[1]为通道数,
        return output

net = LeNet()
print(list(net.parameters())[0].device)
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
# train_iter = train_iter.to(device)#为什么不能在这加
# test_iter = test_iter.to(device)
lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

