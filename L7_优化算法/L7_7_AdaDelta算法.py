#除了RMSProp算法外，AdaDelta算法也针对AdaGrad算法在迭代后期可能较难找到有用解的问题做了改进
#AdaDelta算法没有学习率这一超参数，引入一个新的参数rho
import torch
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

features, labels = d2l.get_data_ch7()
d2l.train_pytorch_ch7(torch.optim.Adadelta, {'rho': 0.9}, features, labels)
