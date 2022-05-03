import torch.nn as nn
from TSM_UResNet import resnet34,resnet50,ResNet


class VideoModel(nn.Module):

    def __init__(self, is_train):
        super().__init__()

        self.train = is_train
        self.ResNet_G = resnet50(type = 'global')
        self.ResNet_L = resnet50(type = 'local')  # resnet不载入预训练数据


    def forward(self, input):

        if self.is_train == 'True' :
            Videoinput1, Videoinput2 = input[0], input[1]
            global_feature  = self.ResNet_G(Videoinput1)
            local_feature = self.ResNet_L(Videoinput2)

            return global_feature, local_feature
        else :
            Videoinput = input
            local_feature = self.ResNet_L(Videoinput)

            return local_feature


