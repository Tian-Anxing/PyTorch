import torch
import torchvision
import numpy as np
import sys
sys.path.append("../..") # 为了导入上层目录的d2lzh_pytorch
import d2lzh_pytorch as d2l

def net(X):
    return d2l.softmax(torch.mm(X.view((-1, num_inputs)), W) + b)
#获取和读取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
#初始化模型参数
num_inputs = 784
num_outputs = 10
W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)
W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

# X = torch.rand((2, 5))
# X_prob = d2l.softmax(X)
# print(X_prob, X_prob.sum(dim=1))
num_epochs, lr = 5, 0.1
d2l.train_ch3(net, train_iter, test_iter, d2l.cross_entropy, num_epochs, batch_size, [W, b], lr)
# 上面的net函数不传参  如果想传参怎么办

#进行预测
X, y = iter(test_iter).__next__()
true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
d2l.show_fashion_mnist(X[0:9], titles[0:9])