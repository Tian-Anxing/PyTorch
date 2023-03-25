#微调由以下四步构成
#1.在源数据集（如ImageNet数据集）上预训练一个神经网络模型，即源模型。
#2.创建一个新的神经网络模型，即目标模型。它复制了源模型上除了输出层外的所有模型设计及其参数。
#我们假设这些模型参数包含了源数据集上学习到的知识，且这些知识同样适用于目标数据集。
#我们还假设源模型的输出层跟源数据集的标签紧密相关，因此在目标模型中不予采用。
#3.为目标模型添加一个输出大小为目标数据集类别个数的输出层，并随机初始化该层的模型参数。
#4.在目标数据集（如椅子数据集）上训练目标模型。我们将从头训练输出层，而其余层的参数都是基于源模型的参数微调得到的。
import matplotlib.pyplot as plt
#当目标数据集远小于源数据集时，微调有助于提升模型的泛化能力。

#热狗识别
#我们将基于一个小数据集对在ImageNet数据集上训练好的ResNet模型进行微调
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models
import os
import sys
import d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#获取数据，data_dir为图片文件夹位置
data_dir = '../data_dir'
os.listdir(os.path.join(data_dir, "hotdog")) # ['train', 'test']
#两个实例分别读取训练数据集和测试数据集
train_imgs = ImageFolder(os.path.join(data_dir, 'hotdog/train'))
test_imgs = ImageFolder(os.path.join(data_dir, 'hotdog/test'))
#画出正例前八和反例后八,ImageFolder获得的是一个包含元组的列表，
#例[('./data/dogcat_2/cat/cat.12484.jpg', 0), ('./data/dogcat_2/cat/cat.12485.jpg', 0)]
#所以train_imgs[i][0]获取的是图片路径
hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4);
plt.show()
#在训练时，我们先从图像中裁剪出随机大小和随机高宽比的一块随机区域，然后将该区域缩放为高和宽均为224像素的输入。
#测试时，我们将图像的高和宽均缩放为256像素，然后从中裁剪出高和宽均为224像素的中心区域作为输入。
#此外，我们对RGB（红、绿、蓝）三个颜色通道的数值做标准化：每个数值减去该通道所有数值的平均值，再除以该通道所有数值的标准差作为输出。

#在使用预训练模型时，一定要和预训练时作同样的预处理。 如果你使用的是torchvision的models，那就要求:
# All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
# 指定RGB三个通道的均值和方差来将图像通道归一化
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_augs = transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

test_augs = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        normalize
    ])
#我们使用在ImageNet数据集上预训练的ResNet-18作为源模型。
#这里指定来自动下载并加载预训练的模型参数。在第一次使用时需要联网下载模型参数。pretrained=True
pretrained_net = models.resnet18(pretrained=True)
# print(pretrained_net)
# print(list(pretrained_net.layer1)[0].conv1)
#改变最后一层的输出个数来满足我们的判断要求，改变后此层就被随机化了，但其他层依然保存着预训练得到的参数，因此一般只需使用较小的学习率来微调这些参数
pretrained_net.fc = nn.Linear(512, 2)
print(pretrained_net.fc)
#PyTorch可以方便的对模型的不同部分设置不同的学习参数，我们在下面代码中将的学习率设为已经预训练过的部分的10倍。
output_params = list(map(id, pretrained_net.fc.parameters()))
feature_params = filter(lambda p: id(p) not in output_params, pretrained_net.parameters())
print(output_params)
print(feature_params)
lr = 0.01
optimizer = optim.SGD([{'params': feature_params},
                       {'params': pretrained_net.fc.parameters(), 'lr': lr * 10}],
                       lr=lr, weight_decay=0.001)
#微调模型
def train_fine_tuning(net, optimizer, batch_size=128, num_epochs=5):
    train_iter = DataLoader(ImageFolder(os.path.join(data_dir, 'hotdog/train'), transform=train_augs),
                            batch_size, shuffle=True)
    test_iter = DataLoader(ImageFolder(os.path.join(data_dir, 'hotdog/test'), transform=test_augs),
                           batch_size)
    loss = torch.nn.CrossEntropyLoss()
    d2l.train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)

train_fine_tuning(pretrained_net, optimizer)
#作为对比定义一个相同的模型，但将它的所有模型参数都初始化为随机值
scratch_net = models.resnet18(pretrained=False, num_classes=2)#num_classes输出种类？
lr = 0.1
optimizer = optim.SGD(scratch_net.parameters(), lr=lr, weight_decay=0.001)
train_fine_tuning(scratch_net, optimizer)