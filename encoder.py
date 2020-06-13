import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from DropBlock import DropBlock2D


class Resnet_sim(nn.Module):
    def __init__(self, opt):
        super(Resnet_sim, self).__init__()
        self.f = []
        if opt.dataset == 'cifar10':
            for name, module in models.resnet50().named_children():
                if name == 'conv1':
                    module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                    self.f.append(module)
        if opt.dataset == 'stl10':
            for _, module in models.resnet50().named_children():
                if not isinstance(module, nn.Linear):
                    self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        print("Feature extractor:", self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, opt.feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


class Resnet_pair(nn.Module):
    def __init__(self, opt):
        super(Resnet_pair, self).__init__()
        self.opt = opt
        self.f = []
        if opt.dataset == 'stl10':
            for _, module in models.resnet50().named_children():
                if not isinstance(module, nn.Linear):
                    self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        print("Feature extractor:", self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, opt.feature_dim, bias=True))


    def forward(self, x1, x2):
        x1 = self.f(x1)
        feature1 = torch.flatten(x1, start_dim=1)
        out1 = self.g(feature1)
        x2 = self.f(x2)
        feature2 = torch.flatten(x2, start_dim=1)
        out2 = self.g(feature2)
        loss = contrast_loss(out1, out2, self.opt)
        return loss, F.normalize(feature1, dim=-1), F.normalize(feature2, dim=-1)


def contrast_loss(out_1, out_2, opt):
    out = torch.cat([out_1, out_2], dim=0)
    size = out_1.size()[0]
    n = size * 2
    s = torch.matmul(out, torch.transpose(out, 0, 1)) / (
            opt.tem * torch.matmul(torch.norm(out, dim=1).unsqueeze(1), torch.norm(out, dim=1).unsqueeze(0))
    )
    a = torch.diag(s)
    l = -(s.exp() / (s.exp().sum(dim=0) - a.exp())).log()
    l1 = torch.diag(l[0: size, size: 2 * size])
    l2 = torch.diag(l[size: 2 * size, 0: size])
    loss = ((l1 + l2).sum()) / n
    return loss


class Bottleneck(nn.Module):
    def __init__(self, encode_num, inplanes, planes, stride=1, expansion=4):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * expansion, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(planes * expansion)
        self.relu = nn.ReLU(True)
        if encode_num == 0:
            self.dropblock = DropBlock2D(drop_prob=0.02, block_size=7)
        elif encode_num == 3:
            self.dropblock = DropBlock2D(drop_prob=0.1, block_size=3)
        else:
            self.dropblock = DropBlock2D(drop_prob=0.05, block_size=5)
        if stride != 1 or inplanes != expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, expansion * planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(expansion * planes)
            )

    def forward(self, x):
        shortcut = self.shortcut(x) if hasattr(self, "shortcut") else x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropblock(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x += shortcut
        x = self.relu(x)
        return x


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

