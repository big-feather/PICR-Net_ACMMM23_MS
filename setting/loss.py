
import mindspore
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
# from torch import einsum
# from einops import rearrange,  repeat
import mindspore.ops as ops
import numpy as np
from math import exp
from mindspore import dtype as mstype

def _iou(pred, target, size_average=True):
    b = pred.shape[0]
    IoU = 0.0
    for i in range(0, b):
        #  compute the IoU of the foreground
        Iand1 = ops.sum(target[i, :, :, :] * pred[i, :, :, :])
        Ior1 = ops.sum(target[i, :, :, :]) + ops.sum(pred[i, :, :, :]) - Iand1
        IoU1 = Iand1 / Ior1
        # IoU loss is (1-IoU1)
        IoU = IoU + (1 - IoU1)

    return IoU / b


class IOU(nn.Cell):
    def __init__(self, size_average=True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def construct(self, pred, target):
        return _iou(pred, target, self.size_average)


def gaussian(window_size, sigma):
    gauss = mindspore.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)], dtype=mstype.float32)
    gauss=ops.div(gauss, gauss.sum())

    return gauss


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = mindspore.Tensor(_2D_window.broadcast_to((channel, 1, window_size, window_size)), dtype=mstype.float32)
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, pad_mode='pad',padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, pad_mode='pad', padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, pad_mode='pad', padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, pad_mode='pad', padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, pad_mode='pad', padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(nn.Cell):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def construct(self, img1, img2):
        _, channel, _, _ = img1.shape

        window = create_window(self.window_size, channel)

        # if img1.is_cuda:
        #     window = window.cuda(img1.get_device())
        # window = window.type_as(img1)

        self.window = window
        self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)