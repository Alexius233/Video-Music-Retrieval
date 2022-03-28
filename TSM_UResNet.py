import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from Hyperparameters import Hyperparameters as hp


def tsm(tensor, duration, version='zero'):  # 默认补0， 可以改成补被顶掉的
    # tensor [N*T, C, H, W]
    size = tensor.size()
    tensor = tensor.view((-1, duration) + size[1:])  # 给了你第二个维度是什么，第一个自己算， 加上其他的截出来得得数组
    # tensor [N, T, C, H, W]
    pre_tensor, post_tensor, peri_tensor = tensor.split([size[1] // 4,
                                                         size[1] // 4,
                                                         size[1] // 2], dim=2)
    if version == 'zero':
        pre_tensor = F.pad(pre_tensor, (0, 0, 0, 0, 0, 0, 1, 0))[:, :-1, ...]
        post_tensor = F.pad(post_tensor, (0, 0, 0, 0, 0, 0, 0, 1))[:, 1:, ...]
    elif version == 'circulant':
        pre_tensor = torch.cat((pre_tensor[:, -1:, ...],
                                pre_tensor[:, :-1, ...]), dim=1)
        post_tensor = torch.cat((post_tensor[:, 1:, ...],
                                 post_tensor[:, :1, ...]), dim=1)
    else:
        raise ValueError('Unknown TSM version: {}'.format(version))

    out = torch.cat((pre_tensor, post_tensor, peri_tensor), dim=2).view(size[0], size[1] * size[2], size[3],
                                                                        size[4])  # 拼回去
    return out


__all__ = ['ResNet_front', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):  # 2个3 * 3 conv 模块
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = tsm(x, 8, 'zero')

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):  # 1*1conv + 3*3conv + 1*1conv 模块
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = tsm(x, 8, 'zero')
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet_front(nn.Module):  # 输入的图像可以随机遮挡，增强rubust

    def __init__(self, block,layers, zero_init_residual=False):  # layers是[ , , , ]的四个参数的Tuple，表示4个部分的数量参数
        super(ResNet_front, self).__init__()

        self.inplanes = 64
        # 7*7conv + 3*3maxpool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.midconv1 = nn.Conv2d(2048, 512, stride=1)

        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks_nums, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride),
                                       nn.BatchNorm2d(planes * block.expansion),
                                       )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks_nums):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        y1 = x
        x = self.layer2(x)
        y2 = x
        x = self.layer3(x)
        y3 = x
        x = self.layer4(x)
        x = self.midconv1(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)

        return x, y1, y2, y3  # 传回来4个上采样的feature
    # stride=1

def StepConv(num_channels, stride, padding):
    """3x3 convolution with padding"""
    return nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=stride,
                     padding=padding, bias=False)


class ResNet_back(nn.Module):  # 写的是resnet-50
    def __init__(self):  # layers是[ , , , ]的四个参数的Tuple，表示4个部分的数量参数
        super(ResNet_back, self).__init__()

        self.tor = 1e-4

        self.y1 = hp.up_paramter1
        self.y2 = hp.up_paramter2
        self.y3 = hp.up_paramter3
        self.params_w = []  # nn.ParameterList([nn.Parameter(torch.randn(size))
        self.relu = nn.ReLU()

        self.upsample = F.upsample

    def upsampling(self,_input, pre, stride, padding, tor):
        size = _input.size()
        _input = F.upsample((size[0], size[1], size[2] * 2, size[3] * 2), mode='nearest')

        out = []
        w1 = nn.Parameter(torch.randn(size))
        w1 = F.relu(w1)
        self.params_w = nn.ParameterList(w1)

        w2 = nn.Parameter(torch.randn(size))
        w2 = F.relu(w2)
        self.params_w = nn.ParameterList(w2)

        out = ((w1 * _input) + (w2 * pre)) / (w1 + w2 + tor)
        out = StepConv(size[2], stride, padding)

        return out

    def forward(self, x):
        # size = [N*T, C, H, W]
        x = self.upsample(x, self.y1, 1, mode='nearest')
        x = self.upsample(x, self.y2, 1, mode='nearest')
        x = self.upsample(x, self.y3, 1, mode='nearest')

        return x

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_front(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_front(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_front(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_front(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_front(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model