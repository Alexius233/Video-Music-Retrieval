import torch
import torch.nn as nn
from Hyperparameters import Hyperparameters as hp

class vice_audiomodel(nn.Module):
    def __init__(self,nfeature):
        super(vice_audiomodel, self).__init__()
        self.nfeature = nfeature
        
        self.Maxpool = nn.AdaptiveMaxPool1d(hp.Pooling_outsize)
        self.vice_net = nn.Sequential(
            nn.Conv1d(self.nfeature, hp.supplement_transform_features,kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hp.supplement_transform_features, 64,kernel_size=1),
        )

    def forward(self, supplement):
        
        #fusing_feature = self.vice_net(supplement.transpose)
        fusing_feature = self.vice_net(supplement)
        fusing_feature = self.Maxpool(fusing_feature)

        return fusing_feature