from __future__ import  absolute_import
import torch as t
from torch import nn
from torchvision.models import vgg16
from torchvision.ops import RoIPool

from model.region_proposal_network import RegionProposalNetwork
from model.faster_rcnn import FasterRCNN
from utilss import array_tool as at
from utilss.config import opt


def decom_vgg16():
    # 特征提取层作为faster-rcnn的起始特征提取，分类层作为RPN的分类层
    # the 30th layer of features is relu of conv5_3
    if opt.caffe_pretrain:
        model = vgg16(pretrained=False)
        if not opt.load_path:
            model.load_state_dict(t.load(opt.caffe_pretrain_path))
    else:
        model = vgg16(not opt.load_path)

    # vgg网络可以分为3大层，一层是features，一层是avgpool,最后一层是classifier
    features = list(model.features)[:30]
    classifier = model.classifier
    classifier = list(classifier)
    # 将vgg16的前4层冻结，不进行反向传播更新权重。然后将分类层提取出来，如果不进行dropout的话就删除dropout层，并且删除了最后一层。
    # del 删除classifier的最后一层
    del classifier[6]
    # 判断是否使用dropout 如果不用dropout就把dropout层删掉
    if not opt.use_drop:
        del classifier[5]
        del classifier[2]
    # 函数在调用多个参数时，在列表、元组、集合、字典及其他可迭代对象作为实参，并在前面加 *
    classifier = nn.Sequential(*classifier)

    # freeze top4 conv
    # 冻结卷积网络前4层
    # 这里为什么是10而不是4呢
    # 因为冻结的卷积层，从第1层卷积网络到第4层卷积网络，中间还有reLu层、池化层
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False

    # 返回值分别为处理过的VGG16的特征提取层和分类层
    return nn.Sequential(*features), classifier


class FasterRCNNVGG16(FasterRCNN):
    """Faster R-CNN based on VGG-16.
    For descriptions on the interface of this model, please refer to
    :class:`model.faster_rcnn.FasterRCNN`.

    Args:
        n_fg_class (int): The number of classes excluding the background.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.

    """

    feat_stride = 16  # downsample 16x for output of conv5 in vgg16

    def __init__(self,
                 # 缺陷的类别个数
                 n_fg_class=20,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32]
                 ):

        # 分别为处理过的VGG16的特征提取层和分类层
        extractor, classifier = decom_vgg16()

        # 产生<=2000个预测框
        rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
        )
        # 得到128个预测框的偏移量和score的预测
        head = VGG16RoIHead(
            n_class=n_fg_class + 1,
            roi_size=7,
            spatial_scale=(1. / self.feat_stride),
            classifier=classifier
        )

        super(FasterRCNNVGG16, self).__init__(
            extractor,
            rpn,
            head,
        )


class VGG16RoIHead(nn.Module):
    """Faster R-CNN Head for VGG-16 based implementation.
    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.
    
    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
        classifier (nn.Module): Two layer Linear ported from vgg16

    """

    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier):
        # n_class includes the background
        # n_class = 21
        super(VGG16RoIHead, self).__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        # roi_size为roi pooling输出特征图的H,W
        self.roi_size = roi_size
        # spatial_scale为预测框缩放的比例
        # 因为预测框的生成是基于原图的HW生成的，而此时的特征图是经过下采样的，因此要对预测框进行缩放
        self.spatial_scale = spatial_scale
        self.roi = RoIPool( (self.roi_size, self.roi_size),self.spatial_scale)

    def forward(self, x, rois, roi_indices):
        """Forward the chain.

        We assume that there are :math:`N` batches.

        Args:
            x (Variable): 4D image variable.
            rois (Tensor): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (Tensor): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.

        """
        # in case roi_indices is  ndarray
        # rois.shape==>[128,4] 建议框
        # roi_indices==>[128] 每个建议框所对应的图像索引 因为每张图像产生128个建议框 所以这128个框对应的图像索引相同
        # rois是rpn层的输出结果
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        # 对两个数组进行拼接 torch.cat
        # indices_and_rois ==>[[score,loc(4个值)]]
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy
        # 现在坐标顺序是 xmin,ymin,xmax,ymax
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        # contiguous函数使tensor变量在内存中的存储变得连续
        indices_and_rois =  xy_indices_and_rois.contiguous()
        # x的通道数为512 是卷积后的特征图
        # indices_and_rois是rpn的输出结果进行处理后的
        # self.roi是进行ROI Pooling
        # pool.shape ==> torch.Size([128, 512, 7, 7])
        # x.shape ==> torch.Size([1, 512, 37, 50]) 1是因为bitch_size=1
        # indices_and_rois.shape ==> torch.Size([128, 5])
        # ROIPooling的第二个参数需满足[k, 5]
        pool = self.roi(x, indices_and_rois)
        # pool.shape ==> torch.Size([128, 25088])
        pool = pool.view(pool.size(0), -1)
        # fc7.shape ==> torch.Size([128, 4096])
        fc7 = self.classifier(pool)
        # roi_cls_locs是对128个anchor偏移量的预测
        roi_cls_locs = self.cls_loc(fc7)
        # roi_scores 是对128个anchor分数的预测
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores


def normal_init(m, mean, stddev, truncated=False):
    # 对VGG16RoIHead网络的全连接层权重初始化过程中，按照图像是否为truncated(截断)分了两种初始化分方法
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
