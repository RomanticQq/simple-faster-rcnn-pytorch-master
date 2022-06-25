import numpy as np
from torch.nn import functional as F
import torch as t
from torch import nn
import numpy as xp

from model.utils.bbox_tools import generate_anchor_base
from model.utils.creator_tool import ProposalCreator

# 从所有的anchor中选出2000个
class RegionProposalNetwork(nn.Module):

    """Region Proposal Network introduced in Faster R-CNN.

    This is Region Proposal Network introduced in Faster R-CNN [#]_.
    This takes features extracted from images and propose
    class agnostic bounding boxes around "objects".

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        in_channels (int): The channel size of input.
        mid_channels (int): The channel size of the intermediate tensor.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.
        feat_stride (int): Stride size after extracting features from an
            image.
        initialW (callable): Initial weight value. If :obj:`None` then this
            function uses Gaussian distribution scaled by 0.1 to
            initialize weight.
            May also be a callable that takes an array and edits its values.
        proposal_creator_params (dict): Key valued paramters for
            :class:`model.utils.creator_tools.ProposalCreator`.

    .. seealso::
        :class:`~model.utils.creator_tools.ProposalCreator`

    """

    def __init__(
            self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2],
            anchor_scales=[8, 16, 32], feat_stride=16,
            proposal_creator_params=dict(),
    ):
        super(RegionProposalNetwork, self).__init__()
        # 获取生成的anchor框
        self.anchor_base = generate_anchor_base(
            anchor_scales=anchor_scales, ratios=ratios)
        # 下采样的步长
        self.feat_stride = feat_stride
        # 实例化ProposalCreator类 该类的功能是返回经过nms处理后的建议框
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)
        # 初试生成的anchor的数量
        n_anchor = self.anchor_base.shape[0]
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        # 因为每一个anchor有两个score，所以要 * 2
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        # 因为每一个anchor的坐标需要4个值来确定
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)
        # 对参数进行初试化
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
        """Forward Region Proposal Network.

        Here are notations.

        * :math:`N` is batch size.
        * :math:`C` channel size of the input.
        * :math:`H` and :math:`W` are height and witdh of the input feature.
        * :math:`A` is number of anchors assigned to each pixel.

        Args:
            x (~torch.autograd.Variable): The Features extracted from images.
                Its shape is :math:`(N, C, H, W)`.
            img_size (tuple of ints): A tuple :obj:`height, width`,
                which contains image size after scaling.
            scale (float): The amount of scaling done to the input images after
                reading them from files.

        Returns:
            (~torch.autograd.Variable, ~torch.autograd.Variable, array, array, array):

            This is a tuple of five following values.

            * **rpn_locs**: Predicted bounding box offsets and scales for \
                anchors. Its shape is :math:`(N, H W A, 4)`.
            * **rpn_scores**:  Predicted foreground scores for \
                anchors. Its shape is :math:`(N, H W A, 2)`.
            * **rois**: A bounding box array containing coordinates of \
                proposal boxes.  This is a concatenation of bounding box \
                arrays from multiple images in the batch. \
                Its shape is :math:`(R', 4)`. Given :math:`R_i` predicted \
                bounding boxes from the :math:`i` th image, \
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            * **roi_indices**: An array containing indices of images to \
                which RoIs correspond to. Its shape is :math:`(R',)`.
            * **anchor**: Coordinates of enumerated shifted anchors. \
                Its shape is :math:`(H W A, 4)`.

        """
        # (N, C, H, W)
        n, _, hh, ww = x.shape
        # self.anchor_base 初试化的anchor框
        # self.feat_stride 下采样的步长
        # hh ww 特征图的高和宽
        # 获取一张图上所有的anchor
        anchor = _enumerate_shifted_anchor(
            np.array(self.anchor_base),
            self.feat_stride, hh, ww)

        # n_anchor 一个像素点产生的anchor数量
        n_anchor = anchor.shape[0] // (hh * ww)
        h = F.relu(self.conv1(x))

        rpn_locs = self.loc(h)
        # UNNOTE: check whether need contiguous
        # A: Yes
        # rpn_locs的shape(N,C,H,W)
        # permute是pytorch交换维度的方法
        # 在PyTorch中，有一些对Tensor的操作不会真正改变Tensor的内容，改变的仅仅是Tensor中字节位置的索引。这些操作有：
        # narrow()，view()，expand()和transpose()
        # 在运行这几个函数之前，需要先把Variable进行.contiguous()操作。
        # 这是因为：有些tensor并不是占用一整块内存，而是由不同的数据块组成，而tensor的view()操作依赖于内存是整块的，这时只需要执行contiguous()这个函数，把tensor变成在内存中连续分布的形式。
        # 例如执行view操作之后，不会开辟新的内存空间来存放处理之后的数据，实际上新数据与原始数据共享同一块内存。
        # 而在调用contiguous()之后，PyTorch会开辟一块新的内存空间存放变换之后的数据，并会真正改变Tensor的内容，按照变换之后的顺序存放数据。
        # n:图片的个数
        # rpn_locs.shape [图片的个数，anchor的个数，4]
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        rpn_scores = self.score(h)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()
        rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4)
        # 此时rpn_softmax_scores的最后一维是两个数 分别为neg和pos 这里的1代表去pos
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()
        # rpn_fg_scores 此时shape为[图片的张数，anchor的数量(anchor的pos-score)](下面的一行代码)
        rpn_fg_scores = rpn_fg_scores.view(n, -1)
        # rpn_scores 的shape为[图片的张数，anchor的数量，neg和pos的score]
        rpn_scores = rpn_scores.view(n, -1, 2)

        rois = list()
        roi_indices = list()
        for i in range(n):
            # 得到经过nms抑制过后的预测框
            roi = self.proposal_layer(
                rpn_locs[i].cpu().data.numpy(),
                rpn_fg_scores[i].cpu().data.numpy(),
                anchor, img_size,
                scale=scale)
            # batch_index 生成anchor所在图片的索引
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)

        # concatenate用于数组的拼接 这里只有rois这一个数组  因为rois是二维的
        # rois 该batch下所有图片的处理过后的anchor
        rois = np.concatenate(rois, axis=0)
        roi_indices = np.concatenate(roi_indices, axis=0)
        # rpn_locs 该batch所有anchor的预测位置偏移量
        # rpn_scores 该batch该所有anchor的score(neg和pos)
        # rois 该batch下所有图片的处理过后的anchor(此时anchor的数量<=2000) 因为要求的是取2000个，但是当剩余的不足2000个的时候就会小于2000
        # roi_indices 该batch下，处理过后每个anchor所对应的图像的索引(此时anchor的数量<=2000)
        # anchor 一张图像上初试产生的所有的anchor
        # print('-'*10)
        # print(rpn_locs.shape)
        # print(rpn_scores.shape)
        # print(rois.shape)
        # print(roi_indices.shape)
        # print(anchor.shape)
        # print('-'*10)
        return rpn_locs, rpn_scores, rois, roi_indices, anchor


def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    # 该函数是在anchor_base的基础上生成所有anchor的过程
    # 因为在anchor_base上生成的9个anchor是在坐标为0,0的基础上生成的
    # 因为每个像素点都要生成9个anchor，对于其他像素点生成anchor的过程可以直接看成是在源点像素点生成的anchor进行平移得到的
    # 细节：产生anchor的数量是以下采样后的特征图的大小为依据的，而anchor的大小是以原图为参照的
    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    # return (K*A, 4)

    # !TODO: add support for torch.CudaTensor
    # xp = cuda.get_array_module(anchor_base)
    # it seems that it can't be boosed using GPU
    import numpy as xp
    # 生成在原图上一系列的纵坐标
    shift_y = xp.arange(0, height * feat_stride, feat_stride)
    # 生成在原图上一系列的横坐标
    shift_x = xp.arange(0, width * feat_stride, feat_stride)
    # np.meshgrid
    # a = np.array([1,2,3])
    # b = np.array([1,2])
    # c, d = np.meshgrid(a, b)
    # a，b值先生成下列这个数组，然后再分别把值赋给c,d
    # [ [1,1],[2,1],[3,1]
    #   [1,2],[2,2],[3,3] ]
    # print(c)
    # [[1 2 3]
    #  [1 2 3]]
    # print(d)
    # [[1 1 1]
    #  [2 2 2]]
    shift_x, shift_y = xp.meshgrid(shift_x, shift_y)
    # ravel把数组变成一位数组
    # axis=1 在每个数组的相同位置取一个数  打包生成一个新数组
    shift = xp.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)
    # 获取anchor_base的数量
    A = anchor_base.shape[0]
    # 获取平移种类的数量
    K = shift.shape[0]
    # 这个地方两个数组相加的意义其实是对anchor_base进行平移，获得所有的anchor
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)

    return anchor


def _enumerate_shifted_anchor_torch(anchor_base, feat_stride, height, width):
    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    # return (K*A, 4)

    # !TODO: add support for torch.CudaTensor
    # xp = cuda.get_array_module(anchor_base)
    import torch as t
    shift_y = t.arange(0, height * feat_stride, feat_stride)
    shift_x = t.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = xp.meshgrid(shift_x, shift_y)
    shift = xp.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        # ???截断正态分布
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        # m.weight.data是卷积核参数, m.bias.data是偏置项参数
        # mean:均值   stddev:方差
        m.weight.data.normal_(mean, stddev)
        # 偏置参数为0
        m.bias.data.zero_()
