import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from DropBlock import DropBlock2D


class ResNet(nn.Module):
    def __init__(self, opt, block=Bottleneck, input_dims=3):
        super(ResNet, self).__init__()
        self.expansion = 4
        self.opt = opt
        self.block = block
        self.inplanes = [64, 256, 512, 1024]
        self.channels = [64, 128, 256, 512]
        self.num_blocks = [3, 4, 6, 3]
        self.encode_num = range(4)
        self.model = nn.Sequential()
        if opt.dataset == 'cifar10':
            self.layer0 = nn.Sequential(
                nn.Conv2d(input_dims, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True)
            )
        elif opt.dataset == 'stl10':
            self.layer0 = nn.Sequential(
                nn.Conv2d(input_dims, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
            )
        for i in self.encode_num:
            if i == 0:
                self.model.add_module("conv1", self.layer0)
                self.model.add_module(
                    "conv {}".format(i),
                    self._make_layer(
                        self.block,
                        i,
                        self.inplanes[i],
                        self.channels[i],
                        self.num_blocks[i],
                        stride=1
                    )
                )
            else:
                self.model.add_module(
                    "conv {}".format(i),
                    self._make_layer(
                        self.block,
                        i,
                        self.inplanes[i],
                        self.channels[i],
                        self.num_blocks[i],
                        stride=2
                    )
                )
        self.last = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # print(self.model)
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, opt.feature_dim, bias=True))

    def _make_layer(self, block, encode_num, inplanes, planes, blocks, stride, expansion=4):
        layers = []
        layers.append(block(encode_num, inplanes, planes, stride, expansion))
        for i in range(1, blocks):
            layers.append(block(encode_num, planes * expansion, planes, expansion=expansion))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        x = self.last(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)

