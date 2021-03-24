import time
from itertools import product as product
from math import ceil
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.models._utils as _utils
from torch.autograd import Variable


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc: torch.Tensor, priors: torch.Tensor, variances: List[float]):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_batch(loc: torch.Tensor, priors: torch.Tensor, variances: List[float]):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    (Batched version)
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [batch_size,num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    batch_size = loc.shape[0]
    batched_priors = priors.unsqueeze(0).repeat(batch_size, 1, 1)
    boxes = torch.cat((
        batched_priors[:, :, :2] + loc[:, :, :2] * variances[0] * batched_priors[:, :, 2:],
        batched_priors[:, :, 2:] * torch.exp(loc[:, :, 2:] * variances[1])), dim=2) # NOTE: Might be the dim need to be 2 ?
    boxes[:, :, :2] -= boxes[:, :, 2:] / 2
    boxes[:, :, 2:] += boxes[:, :, :2]
    return boxes


def decode_landm(pre: torch.Tensor, priors: torch.Tensor, variances: List[float]):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    # NOTE: Check the shape of this to see if we need to change the dim for concat op
    # in batched mode. -> It does
    landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ), dim=1)
    return landms


def decode_landm_batch(pre: torch.Tensor, priors: torch.Tensor, variances: List[float]):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    (Batched version)
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [batch_size,num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    batch_size = pre.shape[0]
    batched_priors = priors.unsqueeze(0).repeat(batch_size, 1, 1)
    landms = torch.cat((batched_priors[:, :, :2] + pre[:, :, :2] * variances[0] * batched_priors[:, :, 2:],
                        batched_priors[:, :, :2] + pre[:, :, 2:4] * variances[0] * batched_priors[:, :, 2:],
                        batched_priors[:, :, :2] + pre[:, :, 4:6] * variances[0] * batched_priors[:, :, 2:],
                        batched_priors[:, :, :2] + pre[:, :, 6:8] * variances[0] * batched_priors[:, :, 2:],
                        batched_priors[:, :, :2] + pre[:, :, 8:10] * variances[0] * batched_priors[:, :, 2:],
                        ), dim=2)
    return landms


def apply_indices_2d_to_3d(idx: torch.Tensor, _3d_tensor: torch.Tensor):
    nLast3Dim, nLast2Dim, nLastDim = _3d_tensor.shape[-3:]
    lastDimCounter = torch.arange(0, nLastDim, dtype=torch.long, device=idx.device)
    last3DimCounter = torch.arange(0, nLast3Dim, dtype=torch.long, device=idx.device)
    return _3d_tensor.reshape(-1)[(idx*nLastDim+(last3DimCounter*nLastDim*nLast2Dim).unsqueeze(-1)
                                   ).unsqueeze(-1).expand(-1, -1, nLastDim) + lastDimCounter]

def py_cpu_nms(dets: torch.Tensor, thresh: float):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # order = scores.argsort()[::-1]
    # NOTE: This line has not been verified to work yet, check it if we get futher roadbloack
    order = torch.argsort(scores, descending=True)

    keep: List[int] = []
    while torch.numel(order) > 0:
        i: int = int(order[0].item())
        keep.append(i)

        # xx1 = np.maximum(x1[i], x1[order[1:]])
        # yy1 = np.maximum(y1[i], y1[order[1:]])
        # xx2 = np.minimum(x2[i], x2[order[1:]])
        # yy2 = np.minimum(y2[i], y2[order[1:]])
        a = x1[i]
        b = x1[order[1:]]
        xx1 = torch.where(a > b, a, b)
        a = y1[i]
        b = y1[order[1:]]
        yy1 = torch.where(a > b, a, b)
        a = x2[i]
        b = x2[order[1:]]
        xx2 = torch.where(a < b, a, b)
        a = y2[i]
        b = y2[order[1:]]
        yy2 = torch.where(a < b, a, b)

        # w = np.maximum(0.0, xx2 - xx1 + 1)
        # h = np.maximum(0.0, yy2 - yy1 + 1)
        b = xx2 - xx1 + 1
        a = torch.zeros_like(b)
        w = torch.where(a > b, a, b)
        b = yy2 - yy1 + 1
        a = torch.zeros_like(b)
        h = torch.where(a > b, a, b)

        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # inds = np.where(ovr <= thresh)[0]
        inds = torch.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

@torch.jit.script
class PriorBox(object):
    def __init__(self, 
                 min_sizes: List[List[int]], 
                 steps: List[int], 
                 clip: bool, 
                 image_size: Tuple[int, int]):
        # super(PriorBox, self).__init__()
        self.min_sizes: List[List[int]] = min_sizes
        self.steps: List[int] = steps
        self.clip: bool = clip
        self.image_size: Tuple[int, int] = image_size
        self.feature_maps: List[List[int]] = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name: str = "s"

    def forward(self):
        anchors: List[float] = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            # for i, j in product(range(f[0]), range(f[1])):
            # NOTE: Since itertools.product does not supported by TorchScript,
            # I will write the nested loop here instead
            for i in range(f[0]):
                for j in range(f[1]):
                    for min_size in min_sizes:
                        s_kx = min_size / self.image_size[1]
                        s_ky = min_size / self.image_size[0]
                        dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                        dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                        # for cy, cx in product(dense_cy, dense_cx):
                        # NOTE: Same as above
                        for cy in dense_cy:
                            for cx in dense_cx:
                                anchors += [cx, cy, s_kx, s_ky]
        # back to torch land
        output = torch.tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output

def conv_bn(inp, oup, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )

def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),
    )

class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel//2, stride=1)

        self.conv5X5_1 = conv_bn(in_channel, out_channel//4, stride=1, leaky = leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

        self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, stride=1, leaky = leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out

class FPN(nn.Module):
    def __init__(self,in_channels_list,out_channels):
        super(FPN,self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride = 1, leaky = leaky)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride = 1, leaky = leaky)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride = 1, leaky = leaky)

        self.merge1 = conv_bn(out_channels, out_channels, leaky = leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky = leaky)

    def forward(self, input: Dict[str, torch.Tensor]):
        input = list(input.values())

        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        return out



class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 8, 2, leaky = 0.1),    # 3
            conv_dw(8, 16, 1),   # 7
            conv_dw(16, 32, 2),  # 11
            conv_dw(32, 32, 1),  # 19
            conv_dw(32, 64, 2),  # 27
            conv_dw(64, 64, 1),  # 43
        )
        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2),  # 43 + 16 = 59
            conv_dw(128, 128, 1), # 59 + 32 = 91
            conv_dw(128, 128, 1), # 91 + 32 = 123
            conv_dw(128, 128, 1), # 123 + 32 = 155
            conv_dw(128, 128, 1), # 155 + 32 = 187
            conv_dw(128, 128, 1), # 187 + 32 = 219
        )
        self.stage3 = nn.Sequential(
            conv_dw(128, 256, 2), # 219 +3 2 = 241
            conv_dw(256, 256, 1), # 241 + 64 = 301
        )
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x
