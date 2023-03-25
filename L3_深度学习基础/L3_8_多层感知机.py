#激活函数
#ReLU(x)=max(x,0).
import torch
import numpy as np
import matplotlib.pylab as plt
import sys
sys.path.append("../..")
import d2lzh_pytorch as d2l

def xyplot(x_vals, y_vals, name):
    d2l.set_figsize(figsize=(5, 2.5))
    d2l.plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy())
    d2l.plt.xlabel('x')
    d2l.plt.ylabel(name + '(x)')

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = x.relu()
xyplot(x, y, 'relu')
plt.show()
y.sum().backward()
xyplot(x, x.grad, 'grad of relu')
plt.show()

# sigmoid函数
y = x.sigmoid()
xyplot(x, y, 'sigmoid')
plt.show()

x.grad.zero_()
y.sum().backward()
xyplot(x, x.grad, 'grad of sigmoid')
plt.show()

# 函数
y = x.tanh()
xyplot(x, y, 'tanh')
plt.show()

x.grad.zero_()
y.sum().backward()
xyplot(x, x.grad, 'grad of tanh')
plt.show()