import torch.nn as nn
from TSM_UResNet import resnet50, ResNet_back


class VideoModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.ResNet_front = resnet50(pretrained=True)  # resnet50载入了imagenet的预训练数据
        self.ResNet_back  = ResNet_back()

    def forward(self, input):
        Videoinput = input
        mid_feature = self.ResNet_front(Videoinput)
        video_feature = self.ResNet_back(mid_feature)

        return  mid_feature,video_feature