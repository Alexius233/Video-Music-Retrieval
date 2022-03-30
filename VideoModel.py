import torch.nn as nn
from TSM_UResNet import resnet34, ResNet_back


class VideoModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.ResNet_front = resnet34()  # resnet不载入预训练数据
        self.ResNet_back  = ResNet_back()

    def forward(self, input):
        Videoinput = input
        mid_feature, upsamplefeature1, upsamplefeature2, upsamplefeature3 = self.ResNet_front(Videoinput)
        video_feature = self.ResNet_back(mid_feature, upsamplefeature1, upsamplefeature2, upsamplefeature3)

        return  mid_feature,video_feature