import torch.nn as nn
from AudioModel import Audiomodel as AMo
from VideoModel import VideoModel as VMo

class TotalModel(nn.Module):
    def __init__(self, n_features):
        super(TotalModel, self).__init__()

        self.videoencoder = VMo()
        self.audioencoder = AMo(dropout=0.5)

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, n_features, bias=False),
        )

    def forward(self, mels, video):

        l_v, h_v = self.videoencoder(video)
        l_a, h_a = self.audioencoder(mels)

        z_v = self.projector(h_v)
        z_a = self.projector(h_a)

        return l_a, l_v, z_a, z_v
