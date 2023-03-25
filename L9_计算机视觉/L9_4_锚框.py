import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import torch

import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
print(torch.__version__) # 1.2.0

#我们通常只对包含s1或r1的大小与宽高比的组合感兴趣，也就是说，以相同像素为中心的锚框的数量为n+m−1，对于整个输入图像，我们将一共生成wh(n+m−1)个锚框。
#以上生成锚框的方法实现在下面的MultiBoxPrior函数中。指定输入、一组大小和一组宽高比，该函数将返回输入的所有锚框。
d2l.set_figsize()
img = Image.open('../catdog.jpg')
w, h = img.size
# print("w = %d, h = %d" % (w, h)) # w = 728, h = 561

# 本函数已保存在d2lzh_pytorch包中方便以后使用
def MultiBoxPrior(feature_map, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5]):
    """
    # 按照「9.4.1. 生成多个锚框」所讲的实现, anchor表示成(xmin, ymin, xmax, ymax).
    https://zh.d2l.ai/chapter_computer-vision/anchor.html
    Args:
        feature_map: torch tensor, Shape: [N, C, H, W].N批量大小，C通道数，H，W高宽
        sizes: List of sizes (0~1) of generated MultiBoxPriores.设置锚框大小0到1之间的数列
        ratios: List of aspect ratios (non-negative) of generated MultiBoxPriores.设置锚框的高宽比
    Returns:
        anchors of shape (1, num_anchors, 4). 由于batch里每个都一样, 所以第一维为1，第二维为锚框的个数，
    """
    #锚框大小形状构造组合
    pairs = [] # pair of (size, sqrt(ration))
    for r in ratios:
        pairs.append([sizes[0], math.sqrt(r)])
    for s in sizes[1:]:
        pairs.append([s, math.sqrt(ratios[0])])
    pairs = np.array(pairs)
    #ss1为锚框的宽，ss2为锚框的高
    ss1 = pairs[:, 0] * pairs[:, 1] # size * sqrt(ration)
    ss2 = pairs[:, 0] / pairs[:, 1] # size / sqrt(ration)
    #形成框与图像的比例
    base_anchors = np.stack([-ss1, -ss2, ss1, ss2],axis=1) / 2#需要使用axis不然会形成4*5的数组
    # print(base_anchors.shape)
    h, w = feature_map.shape[-2:]
    print("h",h)
    shifts_x = np.arange(0, w) / w#从0到w
    shifts_y = np.arange(0, h) / h#从0到h
    shift_x, shift_y = np.meshgrid(shifts_x, shifts_y)#组成网格
    #shift_x中包含h个从0到1的数组，同理shift_y中包含h个相同数字的数组
    shift_x = shift_x.reshape(-1)#降成一维
    shift_y = shift_y.reshape(-1)
    #组成锚点
    shifts = np.stack((shift_x, shift_y, shift_x, shift_y), axis=1)
    # print(shifts.shape)
    anchors = shifts.reshape((-1, 1, 4)) + base_anchors.reshape((1, -1, 4))#得到左上右下坐标？
    return torch.tensor(anchors, dtype=torch.float32).view(1, -1, 4)#重新塑形

X = torch.Tensor(1, 3, h, w)  # 构造输入数据
print("X",X)
Y = MultiBoxPrior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
# print(Y.shape) # torch.Size([1, 2042040, 4])

#再次将Y塑性为（h,w,5,4）的形状
#我们访问以（250，250）为中心的第一个锚框。4个元素，左上角x,y，右下角x,y，其中x和y轴的坐标值分别已除以图像的宽和高，因此值域均为0和1之间。
boxes = Y.reshape((h, w, 5, 4))
# print(boxes[250, 250, 0, :])# * torch.tensor([w, h, w, h], dtype=torch.float32)
print("357" ,boxes[250, 250, :, :])
#为了描绘图像中以某个像素为中心的所有锚框，我们先定义show_bboxes函数以便在图像上画出多个边界框。
# 本函数已保存在d2lzh_pytorch包中方便以后使用
def show_bboxes(axes, bboxes, labels=None, colors=None):
    #axes,bboxes传入锚框的中心位置,labels,colors
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):#isinstance (a,(str,int,list))    # 是元组中的一个返回 True
            obj = [obj]
        return obj

    labels = _make_list(labels)#labels不为空所以没变
    # print("labels",labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])#colors起初为空，把他赋值成数组
    for i, bbox in enumerate(bboxes):#取bbox的下标和内容
        color = colors[i % len(colors)]#五个锚框，五个颜色
        #将边界框(左上x, 左上y, 右下x, 右下y)格式转换成matplotlib格式：
        rect = d2l.bbox_to_rect(bbox.detach().cpu().numpy(), color)#我们只是想要显示他，不需要进行反传，detach()方法出现了。
        axes.add_patch(rect)#加入到绘图列表？
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'#如果框的颜色为白色文本用黑色，如果框的颜色不是白色文本颜色用白色
            axes.text(rect.xy[0], rect.xy[1], labels[i],#rect.xy[0],rect.xy[1]为锚框左上角的坐标，生成文本
                      va='center', ha='center', fontsize=6, color=text_color,
                      bbox=dict(facecolor=color, lw=0))

# d2l.set_figsize()
# fig = d2l.plt.imshow(img)
# bbox_scale = torch.tensor([[w, h, w, h]], dtype=torch.float32)
# show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,#得到真正的坐标点
#             ['s=0.75, r=1', 's=0.75, r=2', 's=0.55, r=0.5', 's=0.5, r=1', 's=0.25, r=1'])
# plt.show()

#交并比
#Jaccard系数即二者交集大小除以二者并集大小，我们通常将Jaccard系数称为交并比
def compute_intersection(set_1, set_2):#计算交集
    """
    计算anchor之间的交集
    Args:
        set_1: a tensor of dimensions (n1, 4), anchor表示成(xmin, ymin, xmax, ymax)
        set_2: a tensor of dimensions (n2, 4), anchor表示成(xmin, ymin, xmax, ymax)
    Returns:
        intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, shape: (n1, n2)
    """
    # PyTorch auto-broadcasts singleton dimensions
    #选出左上角
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)#torch.max返回最大值，.unsqueeze(0)在最外面加一个维度，.unsqueeze(1)内部元素加一个维度
    # print("嘻嘻",set_1[:, :2].unsqueeze(1))
    # print("哈哈", set_2[:, :2].unsqueeze(0))
    #选出右下角
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2), clamp（）函数的功能将输入input张量每个元素的值压缩到区间 [min,max]
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def compute_jaccard(set_1, set_2):
    """
    计算anchor之间的Jaccard系数(IoU)
    Args:
        set_1: a tensor of dimensions (n1, 4), anchor表示成(xmin, ymin, xmax, ymax)
        set_2: a tensor of dimensions (n2, 4), anchor表示成(xmin, ymin, xmax, ymax)
    Returns:
        Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, shape: (n1, n2)
    """
    # Find intersections
    intersection = compute_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)

#标注训练集的锚框
#例子。为图像中猫和狗定义真实边界框，第一个元素为类别（0为狗，1为猫），剩余4个元素为左上角和右下角的坐标（值域在0到1之间）
bbox_scale = torch.tensor((w, h, w, h), dtype=torch.float32)
ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                            [1, 0.55, 0.2, 0.9, 0.88]])
anchors = torch.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                    [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                    [0.57, 0.3, 0.92, 0.9]])
#
# fig = d2l.plt.imshow(img)
# show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
# show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4']);
# plt.show()

#下面实现MultiBoxTarget函数为锚框标注类别和偏移量。背景类别设为0，并令目标类别的整数索引自加1
def assign_anchor(bb, anchor, jaccard_threshold=0.5):
    """
    # 按照「9.4.1. 生成多个锚框」图9.3所讲为每个anchor分配真实的bb, anchor表示成归一化(xmin, ymin, xmax, ymax).
    https://zh.d2l.ai/chapter_computer-vision/anchor.html
    Args:
        bb: 真实边界框(bounding box), shape:（nb, 4）
        anchor: 待分配的anchor, shape:（na, 4）
        jaccard_threshold: 预先设定的阈值
    Returns:
        assigned_idx: shape: (na, ), 每个anchor分配的真实bb对应的索引, 若未分配任何bb则为-1
    """
    na = anchor.shape[0]#anchor.shape[0]为获得anchor的行数，anchor.shape[1]为获得anchor的列数
    nb = bb.shape[0]
    jaccard = compute_jaccard(anchor, bb).detach().cpu().numpy() # shape: (na, nb)
    assigned_idx = np.ones(na) * -1  # 初始全为-1

    # 先为每个bb分配一个anchor(不要求满足jaccard_threshold)
    jaccard_cp = jaccard.copy()
    for j in range(nb):
        i = np.argmax(jaccard_cp[:, j])
        assigned_idx[i] = j
        jaccard_cp[i, :] = float("-inf") # 赋值为负无穷, 相当于去掉这一行

    # 处理还未被分配的anchor, 要求满足jaccard_threshold
    for i in range(na):
        if assigned_idx[i] == -1:
            j = np.argmax(jaccard[i, :])
            if jaccard[i, j] >= jaccard_threshold:
                assigned_idx[i] = j

    return torch.tensor(assigned_idx, dtype=torch.long)

def xy_to_cxcy(xy):
    """
    将(x_min, y_min, x_max, y_max)形式的anchor转换成(center_x, center_y, w, h)形式的.
    https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py
    Args:
        xy: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    Returns:
        bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h

def MultiBoxTarget(anchor, label):
    """
    # 按照「9.4.1. 生成多个锚框」所讲的实现, anchor表示成归一化(xmin, ymin, xmax, ymax).
    https://zh.d2l.ai/chapter_computer-vision/anchor.html
    Args:
        anchor: torch tensor, 输入的锚框, 一般是通过MultiBoxPrior生成, shape:（1，锚框总数，4）
        label: 真实标签, shape为(bn, 每张图片最多的真实锚框数, 5)
               第二维中，如果给定图片没有这么多锚框, 可以先用-1填充空白, 最后一维中的元素为[类别标签, 四个坐标值]
    Returns:
        列表, [bbox_offset, bbox_mask, cls_labels]
        bbox_offset: 每个锚框的标注偏移量，形状为(bn，锚框总数*4)
        bbox_mask: 形状同bbox_offset, 每个锚框的掩码, 一一对应上面的偏移量, 负类锚框(背景)对应的掩码均为0, 正类锚框的掩码均为1
        cls_labels: 每个锚框的标注类别, 其中0表示为背景, 形状为(bn，锚框总数)
    """
    assert len(anchor.shape) == 3 and len(label.shape) == 3
    bn = label.shape[0]#此bn为批量大小？
    # print("12",label.shape)
    # print(label.shape)

    def MultiBoxTarget_one(anc, lab, eps=1e-6):
        """
        MultiBoxTarget函数的辅助函数, 处理batch中的一个
        Args:
            anc: shape of (锚框总数, 4)
            lab: shape of (真实锚框数, 5), 5代表[类别标签, 四个坐标值]
            eps: 一个极小值, 防止log0
        Returns:
            offset: (锚框总数*4, )
            bbox_mask: (锚框总数*4, ), 0代表背景, 1代表非背景
            cls_labels: (锚框总数, 4), 0代表背景
        """
        print("lab",lab)
        an = anc.shape[0]
        assigned_idx = assign_anchor(lab[:, 1:], anc) # 给真实锚框分配锚框索引,从1开始是因为标签0为类别标签
        # print("hah",assigned_idx)
        bbox_mask = ((assigned_idx >= 0).float().unsqueeze(-1)).repeat(1, 4) # (锚框总数, 4) 掩码变量，背景处为4个0，其他为4个1
        # print("bbox",bbox_mask)
        # print(an)
        cls_labels = torch.zeros(an, dtype=torch.long) # 0表示背景
        print(cls_labels)
        assigned_bb = torch.zeros((an, 4), dtype=torch.float32) # 所有anchor对应的bb坐标
        print(assigned_bb)
        for i in range(an):
            bb_idx = assigned_idx[i]
            if bb_idx >= 0: # 即非背景
                cls_labels[i] = lab[bb_idx, 0].long().item() + 1 # 注意要加一,改变了标签的代表值，背景为0，狗由0为1，猫由1为2
                assigned_bb[i, :] = lab[bb_idx, 1:] # 给锚框匹配真实锚框位置
        # print("cs",cls_labels)
        # print("ab",assigned_bb)
        center_anc = xy_to_cxcy(anc) # (center_x, center_y, w, h)，改变锚框位置格式
        center_assigned_bb = xy_to_cxcy(assigned_bb)# 改变真实锚框位置格式
        print(center_anc)
        print(center_assigned_bb)
        offset_xy = 10.0 * (center_assigned_bb[:, :2] - center_anc[:, :2]) / center_anc[:, 2:]
        offset_wh = 5.0 * torch.log(eps + center_assigned_bb[:, 2:] / center_anc[:, 2:])
        offset = torch.cat([offset_xy, offset_wh], dim = 1) * bbox_mask # (锚框总数, 4)

        return offset.view(-1), bbox_mask.view(-1), cls_labels #.view重新定义数组形状， 返回锚框偏移,掩码，新的锚框匹配标签

    batch_offset = []
    batch_mask = []
    batch_cls_labels = []
    for b in range(bn):
        offset, bbox_mask, cls_labels = MultiBoxTarget_one(anchor[0, :, :], label[b, :, :])

        batch_offset.append(offset)
        batch_mask.append(bbox_mask)
        batch_cls_labels.append(cls_labels)

    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    cls_labels = torch.stack(batch_cls_labels)
    print("bo",bbox_offset)
    return [bbox_offset, bbox_mask, cls_labels] #返回一个批量的锚框偏移,掩码，新的锚框匹配标签的数组

labels = MultiBoxTarget(anchors.unsqueeze(dim=0),
                        ground_truth.unsqueeze(dim=0))

# print(labels[2])
# print(labels[1])
# print(labels[0])

#输出预测边界框
#在模型预测阶段，我们先为图像生成多个锚框，当锚框数量较多时，同一个目标上可能会输出较多相似的预测边界框。
# 为了使结果更加简洁，我们可以移除相似的预测边界框。常用的方法叫作非极大值抑制
#例子，先构造4个锚框。简单起见，我们假设预测偏移量全是0：预测边界框即锚框。最后，我们构造每个类别的预测概率。
anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                        [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
offset_preds = torch.tensor([0.0] * (4 * len(anchors)))
cls_probs = torch.tensor([[0., 0., 0., 0.,],  # 背景的预测概率
                          [0.9, 0.8, 0.7, 0.1],  # 狗的预测概率
                          [0.1, 0.2, 0.3, 0.9]])  # 猫的预测概率

fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, anchors * bbox_scale,
            ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])
plt.show()
# 以下函数已保存在d2lzh_pytorch包中方便以后使用
from collections import namedtuple
Pred_BB_Info = namedtuple("Pred_BB_Info", ["index", "class_id", "confidence", "xyxy"])

def non_max_suppression(bb_info_list, nms_threshold = 0.5):
    """
    非极大抑制处理预测的边界框
    Args:
        bb_info_list: Pred_BB_Info的列表, 包含预测类别、置信度等信息
        nms_threshold: 阈值
    Returns:
        output: Pred_BB_Info的列表, 只保留过滤后的边界框信息
    """
    output = []
    # 先根据置信度从高到低排序
    print("bb,list",bb_info_list)
    sorted_bb_info_list = sorted(bb_info_list, key = lambda x: x.confidence, reverse=True)

    while len(sorted_bb_info_list) != 0:
        best = sorted_bb_info_list.pop(0)
        output.append(best)

        if len(sorted_bb_info_list) == 0:
            break

        bb_xyxy = []
        for bb in sorted_bb_info_list:
            bb_xyxy.append(bb.xyxy)#把剩下的全部加入数组

        iou = compute_jaccard(torch.tensor([best.xyxy]),
                              torch.tensor(bb_xyxy))[0] # shape: (len(sorted_bb_info_list), )
        # print(compute_jaccard(torch.tensor([best.xyxy]),torch.tensor(bb_xyxy)))
        # print("iou",iou)
        n = len(sorted_bb_info_list)
        # print("1",sorted_bb_info_list)
        sorted_bb_info_list = [sorted_bb_info_list[i] for i in range(n) if iou[i] <= nms_threshold]#保留下了标签不相符的锚框,删掉了标签相符且锚框与置信度最大的锚框相似的锚框。
        # print("2",sorted_bb_info_list)
        # print("out",output) #因为还剩下一个与标签不相符的锚框，所以会进入循环，但把他加入到output后会执行break
    return output

def MultiBoxDetection(cls_prob, loc_pred, anchor, nms_threshold = 0.5):
    """
    # 按照「9.4.1. 生成多个锚框」所讲的实现, anchor表示成归一化(xmin, ymin, xmax, ymax).
    https://zh.d2l.ai/chapter_computer-vision/anchor.html
    Args:
        cls_prob: 经过softmax后得到的各个锚框的预测概率, shape:(bn, 预测总类别数+1, 锚框个数)#总类别数加一，加上背景类别
        loc_pred: 预测的各个锚框的偏移量, shape:(bn, 锚框个数*4)
        anchor: MultiBoxPrior输出的默认锚框, shape: (1, 锚框个数, 4)
        nms_threshold: 非极大抑制中的阈值
    Returns:
        所有锚框的信息, shape: (bn, 锚框个数, 6)
        每个锚框信息由[class_id, confidence, xmin, ymin, xmax, ymax]表示
        class_id=-1 表示背景或在非极大值抑制中被移除了
    """
    assert len(cls_prob.shape) == 3 and len(loc_pred.shape) == 2 and len(anchor.shape) == 3
    bn = cls_prob.shape[0]

    def MultiBoxDetection_one(c_p, l_p, anc, nms_threshold = 0.5):
        """
        MultiBoxDetection的辅助函数, 处理batch中的一个
        Args:
            c_p: (预测总类别数+1, 锚框个数)
            l_p: (锚框个数*4, )
            anc: (锚框个数, 4)
            nms_threshold: 非极大抑制中的阈值
        Return:
            output: (锚框个数, 6)
        """
        pred_bb_num = c_p.shape[1]#概率的个数
        anc = (anc + l_p.view(pred_bb_num, 4)).detach().cpu().numpy() # 加上偏移量
        # print("c_p",c_p)
        # print("2",torch.max(c_p, 0),"3")
        confidence, class_id = torch.max(c_p, 0)
        confidence = confidence.detach().cpu().numpy()
        class_id = class_id.detach().cpu().numpy()

        pred_bb_info = [Pred_BB_Info(
                            index = i,
                            class_id = class_id[i] - 1, # 正类label从0开始
                            confidence = confidence[i],
                            xyxy=[*anc[i]]) # xyxy是个列表,在列表前加*号，会将列表拆分成一个一个的独立元素
                        for i in range(pred_bb_num)]
        # print("s",non_max_suppression(pred_bb_info, nms_threshold),"e")
        # 正类的index
        obj_bb_idx = [bb.index for bb in non_max_suppression(pred_bb_info, nms_threshold)]#获得最大概率的标签
        print("obj",obj_bb_idx)
        output = []
        print("dangxia",pred_bb_info)
        for bb in pred_bb_info:
            output.append([
                (bb.class_id if bb.index in obj_bb_idx else -1.0),
                bb.confidence,
                *bb.xyxy
            ])#将非最大置信的锚框标签设置为-1
        print("output",output)
        return torch.tensor(output) # shape: (锚框个数, 6)
    print("anc",anchor[0])
    batch_output = []
    for b in range(bn):
        batch_output.append(MultiBoxDetection_one(cls_prob[b], loc_pred[b], anchor[0], nms_threshold))#批量处理,anchor[0]为去掉最外面那一维

    return torch.stack(batch_output)

output = MultiBoxDetection(
    cls_probs.unsqueeze(dim=0), offset_preds.unsqueeze(dim=0),
    anchors.unsqueeze(dim=0), nms_threshold=0.5)
print(output)
print("xixi",cls_probs.unsqueeze(dim=0))
fig = d2l.plt.imshow(img)#打印出标签不为-1的
for i in output[0].detach().cpu().numpy():
    if i[0] == -1:
        continue
    label = ('dog=', 'cat=')[int(i[0])] + str(i[1])#加号前选择文本，加号后选择置信数值
    print("159",i,torch.tensor(i[2:]) * bbox_scale)
    show_bboxes(fig.axes, [torch.tensor(i[2:]) * bbox_scale], label)#需要加中括号变为数组是因为，需要参数为二维数组，
plt.show()