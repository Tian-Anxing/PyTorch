# 本节我们将使用torchvision包，它是服务于PyTorch深度学习框架的，主要用来构建计算机视觉模型。torchvision主要由以下几部分构成：
#
# torchvision.datasets: 一些加载数据的函数及常用的数据集接口；
# torchvision.models: 包含常用的模型结构（含预训练模型），例如AlexNet、VGG、ResNet等；
# torchvision.transforms: 常用的图片变换，例如裁剪、旋转等；
# torchvision.utils: 其他的一些有用的方法。

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
# sys.path.append("..") # 为了导入上层目录的d2lzh_pytorch
import d2lzh_pytorch as d2l
import torch.utils.data

#获取数据集
mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True, transform=transforms.ToTensor())
#上面的mnist_train和mnist_test都是torch.utils.data.Dataset的子类，所以我们可以用len()来获取该数据集的大小，还可以用下标来获取具体的一个样本。
# print(type(mnist_train))
# print(len(mnist_train), len(mnist_test))
#通过下标访问样本
feature, label = mnist_train[0]
# print(feature.shape, label)
#需要注意的是，feature的尺寸是 (C x H x W)

# X, y = [], []
# for i in range(10):
#     X.append(mnist_train[i][0])
#     y.append(mnist_train[i][1])
# d2l.show_fashion_mnist(X, d2l.get_fashion_mnist_labels(y))
batch_size = 256
if sys.platform.startswith('win'):
    num_workers = 0  # 0表示不用额外的进程来加速读取数据
else:
    num_workers = 4
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
start = time.time()

for X,y in train_iter:
    continue
print("%.2f sec" % (time.time()-start))