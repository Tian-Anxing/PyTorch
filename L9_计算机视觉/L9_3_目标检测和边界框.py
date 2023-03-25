import matplotlib.pyplot as plt
from PIL import Image
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

d2l.set_figsize()
img = Image.open('../catdog.jpg')
d2l.plt.imshow(img);  # 加分号只显示图
plt.show()
#边界框
# bbox是bounding box的缩写，给定位置
dog_bbox, cat_bbox = [60, 45, 378, 516], [400, 112, 655, 493]

def bbox_to_rect(bbox, color):  # 本函数已保存在d2lzh_pytorch中方便以后使用
    # 将边界框(左上x, 左上y, 右下x, 右下y)格式转换成matplotlib格式：
    # ((左上x, 左上y), 宽, 高)
    return d2l.plt.Rectangle(#矩形函数
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=True, edgecolor=color, linewidth=2)
#画出边框
fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'));
plt.show()