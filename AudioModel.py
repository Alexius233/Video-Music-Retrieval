
import torch.nn as nn
from WFN import WPN
from GQDL import GQDL
from Vice_Audionet import vice_audiomodel as Vice
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
        self.Vice = Vice()
        self.dropout = dropout

        self.Bottleneck1 = nn.Conv1d(hp.out2pool, hp.af_dim, kernel_size=1, stride=1)
        self.Bottleneck2 = nn.Conv1d(hp.af_dim, hp.out2pool, kernel_size=1, stride=1)
        self.net = nn.Sequential(
            self.Bottleneck1,
            self.Bottleneck2,
        )

    def forward(self, input):

        Specinput, supplement = input
        mid_feature = self.WFN(Specinput)
        supplement_feature = self.Vice(supplement)
        #原始版融合策略：先加再一个卷积
        mid_feature = mid_feature + supplement_feature
        mid_feature = self.net(mid_feature)
        spec_feature = self.GQDL(mid_feature)


        return  spec_feature
