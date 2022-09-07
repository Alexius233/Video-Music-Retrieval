import torch
import torch.nn as nn
from model.Audio.WFN import WPN
from model.Audio.GQDL import GQDL
from model.Audio.Vice_Audionet import vice_audiomodel as Vice
from Hyperparameters import Hyperparameters as hp


class Audiomodel(nn.Module):
    '''
        input:
            mels: [N, T_y/r, n_mels*r]
        output:
            spec_feature --- [N, H, W]

    '''

    def __init__(self, dropout):
        super().__init__()

        self.WFN = WPN(dropout)
        self.GQDL = GQDL()
        self.Vice = Vice(86) 
        self.dropout = dropout

        self.Bottleneck1 = nn.Conv1d(hp.meltingc, hp.af_dim, kernel_size=1, stride=1)
        self.Bottleneck2 = nn.Conv1d(hp.af_dim, 128, kernel_size=1, stride=1)
        self.meltingnet = nn.Sequential(
            self.Bottleneck1,
            self.Bottleneck2,
        )

    def forward(self, mels, supplement_data):

        Specinput = mels
        supplement = supplement_data
        if torch.isnan(Specinput).sum()>0 or torch.isnan(supplement).sum()>0:
            print("音频输入存在NaN")
        mid_feature = self.WFN(Specinput)
        if torch.isnan(mid_feature).sum()>0:
            print("TCN存在NaN")
        #print("一")
        #print(mid_feature.shape)
        supplement_feature = self.Vice(supplement_data)
        if torch.isnan(supplement_feature).sum()>0:
            print("vice存在NaN")
        #print("二")
        #print(supplement_feature.shape)
        #原始版融合策略：先加再一个卷积（可以改进）
        mid_feature = torch.cat((mid_feature, supplement_feature), 1)
        mid_feature = self.meltingnet(mid_feature)
        if torch.isnan(mid_feature).sum()>0:
            print("融合后存在NaN")
        spec_feature = self.GQDL(mid_feature)
        if torch.isnan(spec_feature).sum()>0:
            print("attention后存在NaN")


        return  spec_feature
