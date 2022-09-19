import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Callable, Any, Optional, Tuple, List


class DeepSim(nn.Module):
    def __init__(self):
        super(DeepSim, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5,5))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2))
        self.naive1 = naiveModule(in_channels=64, out_channels=64)
        self.naive2 = naiveModule(in_channels=256, out_channels=256)
        self.inception_a1 = InceptionA(in_channels=64, pool_features=32)
        self.inception_a2 = InceptionA(in_channels=256, pool_features=32)
        self.fc = nn.Linear(256,256)

    def forward(self,x):
        x = self.conv1(x)
        x = self.maxpool1(x)    #torch.Size([8, 64, 62, 62])
        x = self.naive1(x)      #torch.Size([8, 64, 30, 30])
        x = self.inception_a1(x)#torch.Size([8, 256, 30, 30])
        x = self.naive2(x)
        x = self.inception_a2(x)
        x = self.naive2(x)
        x = self.inception_a2(x)
        x = self.naive2(x)
        x = self.inception_a2(x)
        x = self.naive2(x)
        x = self.inception_a2(x)#torch.Size([8, 256, 1, 1])
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x


class naiveModule(nn.Module):   #naive module
    def __init__(self, in_channels, out_channels):
        super(naiveModule, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=(2,2))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class InceptionA(nn.Module):    #dimension reduction module
    def __init__(
        self,
        in_channels: int,
        pool_features: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionA, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = conv_block(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = conv_block(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, padding=1)

        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class BasicConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs: Any
    ) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)



# x = torch.rand(8,129,128)
# x = x.unsqueeze(1)
# deepSim = DeepSim()
# print(deepSim(x).shape)