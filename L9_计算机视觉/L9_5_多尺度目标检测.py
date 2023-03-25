import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

img = Image.open('../catdog.jpg')
w, h = img.size # (728, 561)
#当使用较小锚框来检测较小目标时，我们可以采样较多的区域；而当使用较大锚框来检测较大目标时，我们可以采样较少的区域。
#我们可以通过定义特征图的形状来确定任一图像上均匀采样的锚框中心。
#display_anchors函数，在特征图fmap上以每个单元（像素）为中心生成锚框anchors
d2l.set_figsize()

def display_anchors(fmap_w, fmap_h, s):
    # 前两维的取值不影响输出结果(原书这里是(1, 10, fmap_w, fmap_h), 我认为错了)
    fmap = torch.zeros((1, 10, fmap_h, fmap_w), dtype=torch.float32)

    # 平移所有锚框使均匀分布在图片上
    offset_x, offset_y = 1.0/fmap_w, 1.0/fmap_h
    # print(d2l.MultiBoxPrior(fmap, sizes=s, ratios=[1, 2, 0.5]).shape)
    anchors = d2l.MultiBoxPrior(fmap, sizes=s, ratios=[1, 2, 0.5]) \
              + torch.tensor([offset_x/2, offset_y/2, offset_x/2, offset_y/2])

    bbox_scale = torch.tensor([[w, h, w, h]], dtype=torch.float32)
    d2l.show_bboxes(d2l.plt.imshow(img).axes,
                    anchors[0] * bbox_scale)
display_anchors(fmap_w=4, fmap_h=2, s=[0.15])#个数4*2
plt.show()
display_anchors(fmap_w=2, fmap_h=1, s=[0.4])#个数2*1
plt.show()
display_anchors(fmap_w=1, fmap_h=1, s=[0.8])#个数1*1
plt.show()