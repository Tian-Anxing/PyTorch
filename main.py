# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import torch
from torch import nn

# a = np.random.random((5,1,4))
# b = np.random.random((1,4,4))
# c = a+b
# print("a",a)
# print("b",b)
# print("c",c)
# print(c.shape)

# a = np.random.random((5,3,4))
# a = a[4,:,:]
# for i,temp in enumerate(a):
#     print(i)

# a = [0,1,2,3]
# if a and len(a)>2:
#     print("hi")
#
# a = [[1,2,3],[4,5,6],[7,8,9]]
# a = torch.Tensor(a)
# b = [[2,2,2]]
# b = torch.Tensor(b)
# c = [2,2,2]
# c = torch.Tensor(c)
# print(a*b)
# print("\n")
# print(a*c)
#
# print(a[:,:1])

# a = [[[0.0000, 0.1000]],
#
#         [[0.1500, 0.2000]],
#
#         [[0.6300, 0.0500]],
#
#         [[0.6600, 0.4500]],
#
#         [[0.5700, 0.3000]]]
# b = [[[0.1000, 0.0800],
#          [0.5500, 0.2000]]]
# a = torch.tensor(a)
# b = torch.tensor(b)
# print(torch.max(a,b))

a = [[[1,2],[3,1],[1,1]]]
a = torch.tensor(a)
print(len(a.shape))
print(a[:,1:])