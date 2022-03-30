import torch.nn.functional as F
import torch.nn as nn
from AudioModel import Audiomodel as AMo
from VideoModel import VideoModel as VMo
from Hyperparameters import Hyperparameters as hp

class TotalModel(nn.Module):
    def __init__(self, dropout, is_train = True):
        super(TotalModel, self).__init__()

        self.videoencoder = VMo()
        self.audioencoder = AMo(dropout=dropout)
        self.is_train = is_train

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        # i decide to transfer the tensor into size:[batch_size, 1024]
        self.projector1 = nn.Sequential(
            nn.Linear(hp.n_features1, hp.n_features, bias=False),
            nn.ReLU(),
        )
        self.projector2 = nn.Sequential(
            nn.Linear(hp.n_features2, hp.n_features, bias=False),
            nn.ReLU(),
        )
    def projector(self, tensor):
        tensor = tensor.view(tensor.size(0), -1)
        linear = nn.Linear(tensor.size(1), 1024)
        tensor = linear(tensor)
        tensor = F.relu(tensor)

        return tensor

    def forward(self, mels, video):

        l_v, h_v = self.videoencoder(video)
        l_a, h_a = self.audioencoder(mels)

        m_v = self.projector(l_v)
        m_a = self.projector(l_a)

        z_v = self.projector(h_v)
        z_a = self.projector(h_a)



        if self.is_train:
            return m_a, m_v, z_a, z_v
        else:
            return h_v, h_a
