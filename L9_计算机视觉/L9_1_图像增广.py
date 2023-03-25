#图像增广通过对训练图像做一系列随机改变，来产生相似但又不同的训练样本，从而扩大训练数据集的规模。
#例如，我们可以对图像进行不同方式的裁剪，使感兴趣的物体出现在不同位置，从而减轻模型对物体出现位置的依赖性。

import time
import torch
import torchvision
import sys
sys.path.append("../")
import d2lzh_pytorch as d2l

from matplotlib import pyplot as plt
from torch import nn,optim
from torch.utils.data import Dataset,DataLoader
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# d2l.set_figsize()
# img = Image.open("../rainier.jpg")
# # plt.imshow(img)
# # plt.show()
#
# # 本函数已保存在d2lzh_pytorch包中方便以后使用
# def show_images(imgs, num_rows, num_cols, scale=2):
#     figsize = (num_cols * scale, num_rows * scale)
#     _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
#     for i in range(num_rows):
#         for j in range(num_cols):
#             axes[i][j].imshow(imgs[i * num_cols + j])
#             axes[i][j].axes.get_xaxis().set_visible(False)
#             axes[i][j].axes.get_yaxis().set_visible(False)
#     return axes
# def apply(img, aug, num_rows=2, num_cols=4, scale=3):
#     Y = [aug(img) for _ in range(num_rows * num_cols)]
#     show_images(Y, num_rows, num_cols, scale)
#
# #翻转和裁剪
# #RandomHorizontalFlip实现一半概率的图像水平（左右）翻转
# apply(img, torchvision.transforms.RandomHorizontalFlip())
# #RandomVerticalFlip实现一半概率的图像垂直（上下）翻转
# apply(img, torchvision.transforms.RandomVerticalFlip())
# #在下面的代码里，我们每次随机裁剪出一块面积为原面积10%∼100%的区域，且该区域的宽和高之比随机取自0.5∼2，然后再将该区域的宽和高分别缩放到200像素。
# #若无特殊说明，本节中a和b之间的随机数指的是从区间[a,b]中随机均匀采样所得到的连续值。
# shape_aug = torchvision.transforms.RandomResizedCrop(200, scale=(0.1, 1), ratio=(0.5, 2))
# apply(img, shape_aug)
# # plt.show()
#
# #变化颜色
# #我们可以从4个方面改变图像的颜色：亮度（brightness）、对比度（contrast）、饱和度（saturation）和色调（hue）。
# #在下面的例子里，我们将图像的亮度随机变化为原图亮度的50%（1−0.5）∼150%（1+0.5）。
# apply(img, torchvision.transforms.ColorJitter(brightness=0.5))
# # plt.show()
# #我们也可以随机变化图像的色调
# apply(img, torchvision.transforms.ColorJitter(hue=0.5))
# #我们也可以随机变化图像的对比度
# apply(img, torchvision.transforms.ColorJitter(contrast=0.5))
# #我们也可以同时设置如何随机变化图像的亮度（brightness）、对比度（contrast）、饱和度（saturation）和色调（hue）
# color_aug = torchvision.transforms.ColorJitter(
#     brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
# apply(img, color_aug)
# # plt.show()
#
# #叠加多个图像增广方法
# augs = torchvision.transforms.Compose([
#     torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
# apply(img, augs)
# # plt.show()
#
# #使用图像增广训练模型
# all_imges = torchvision.datasets.CIFAR10(train=True, root="../CIFAR", download=True)
# show_images([all_imges[i][0] for i in range(32)], 4, 8, scale=0.8);
# plt.show()
#为了在预测时得到确定的结果，我们通常只将图像增广应用在训练样本上，而不在预测时使用含随机操作的图像增广
#此外，我们使用ToTensor将小批量图像转成PyTorch需要的格式，即形状为(批量大小, 通道数, 高, 宽)、值域在0到1之间且类型为32位浮点数。
flip_aug = torchvision.transforms.Compose([
     torchvision.transforms.RandomHorizontalFlip(),
     # torchvision.transforms.Resize(size=96),
     torchvision.transforms.ToTensor()])
no_aug = torchvision.transforms.Compose([
     # torchvision.transforms.Resize(size=96),
     torchvision.transforms.ToTensor()])

#定义一个辅助函数来方便读取图像并应用图像增广。有关DataLoader的详细介绍，可参考更早的3.5节图像分类数据集(Fashion-MNIST)。
num_workers = 0 if sys.platform.startswith('win32') else 4
def load_cifar10(is_train, augs, batch_size, root="../CIFAR"):
    dataset = torchvision.datasets.CIFAR10(root=root, train=is_train, transform=augs, download=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers)
#使用图像增广训练模型
def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = d2l.evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
#然后就可以定义train_with_data_aug函数使用图像增广来训练模型了。
# 该函数使用Adam算法作为训练使用的优化算法，然后将图像增广应用于训练数据集之上，最后调用刚才定义的train函数训练并评价模型。
def train_with_data_aug(train_augs, test_augs, lr=0.001):
    batch_size, net = 256, d2l.resnet18()#该数据是三通道所以需要把resnet18的输入通道改成3
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = torch.nn.CrossEntropyLoss()
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    train(train_iter, test_iter, net, loss, optimizer, device, num_epochs=10)
#训练模型
train_with_data_aug(flip_aug, no_aug)