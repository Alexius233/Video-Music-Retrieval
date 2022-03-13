
import torch.nn as nn
from WFN import WPN
from GQDL import GQDL


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
        self.dropout = dropout

    def forward(self, input):

        Specinput = input
        mid_feature = self.GQDL(Specinput)
        spec_feature = self.GQDL(mid_feature)


        return mid_feature, spec_feature
        # 返回中间特征和最后特征