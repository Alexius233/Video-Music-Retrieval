import torch.nn as nn
from Hyperparameters import Hyperparameters as hp

class vice_audiomodel(nn.Module):
    def __init__(self,nfeature):
        super(vice_audiomodel, self).__init__()
        self.nfeature = nfeature

        self.vice_net = nn.Sequential(
            nn.Linear(self.nfeature, hp.supplement_transform_features, bias=False),
            nn.ReLU(),
            nn.Linear(hp.supplement_transform_features, 256, bias=False),
            nn.AdaptiveMaxPool1d(hp.Pooling_outsize)
        )

    def forward(self, supplement):
        self.nfeature = supplement[1].size
        fusing_feature = self.vice_net(supplement)

        return fusing_feature