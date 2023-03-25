#Adam算法在RMSProp算法基础上对小批量随机梯度也做了指数加权移动平均
#所以Adam算法可以看做是RMSProp算法与动量法的结合。
import torch
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
features, labels = d2l.get_data_ch7()
d2l.train_pytorch_ch7(torch.optim.Adam, {'lr': 0.01}, features, labels)