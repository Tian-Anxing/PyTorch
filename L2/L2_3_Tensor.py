import torch
# #requires_grad = True 追踪gard
# #用with torch.no_grad()
# #        ...不需要被追踪的代码
# x = torch.ones(2,2,requires_grad=True)
# print(x)
# print(x.grad_fn)
# y = x+2
# print(y)
# print(y.grad_fn)
# z = y * y * 3
# out = z.mean()
# print(z, out)
# out.backward()
# print(x.grad)
# #如果out是标量，则不需要为backward()传入任何参数；否则，需要传入一个与out同形的Tensor x.grad 会累加 需要有清0操作

# x = torch.ones(2,2)
# x = ((x * 3) / (x ))
# print(x.requires_grad)
# x.requires_grad_(True)
# print((x.requires_grad))
# b = (x * x).sum()
# print(b)

# x = torch.tensor(1.0, requires_grad=True)
# y1 = x ** 2
# with torch.no_grad():
#     y2 = x ** 3
# y3 = y1 + y2
# print(x.requires_grad)
# print(y1, y1.requires_grad) # True
# print(y2, y2.requires_grad) # False
# print(y3, y3.requires_grad) # True
# y3.backward()
# print(x.grad)

#此外，如果我们想要修改tensor的数值，但是又不希望被autograd记录（即不会影响反向传播），那么我么可以对tensor.data进行操作。
x = torch.ones(1,requires_grad=True)
print(x.data) # 还是一个tensor
print(x.data.requires_grad) # 但是已经是独立于计算图之外
y = 2 * x
x.data *= 100 # 只改变了值，不会记录在计算图，所以不会影响梯度传播
y.data *= 100
y.backward()
print(x) # 更改data的值也会影响tensor的值
print(x.grad)