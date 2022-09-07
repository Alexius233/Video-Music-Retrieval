import torch
import torch.nn as nn
from model.Vision.TSM_UResNet import resnet34,resnet50,ResNet


class VideoModel(nn.Module):

    def __init__(self, is_train):
        super().__init__()

        self.is_train = is_train
        self.ResNet_G = resnet50(duration=8)
        self.ResNet_L = resnet50(duration=64)  # resnet不载入预训练数据


    def forward(self, frames1, frames2):

        if self.is_train == True:
            Videoinput1 = frames1
            Videoinput2 = frames2
            global_feature  = self.ResNet_G(Videoinput1)
            local_feature = self.ResNet_L(Videoinput2)
           

            return global_feature, local_feature

        else:
            Videoinput = inputs
            local_feature = self.ResNet_L(Videoinput)

            return local_feature


