import torch.nn.functional as F
import torch.nn as nn
from AudioModel import Audiomodel as AMo
from VideoModel import VideoModel as VMo
from IndexFeature import Weight
from Hyperparameters import Hyperparameters as hp

class TotalModel(nn.Module):
    def __init__(self, is_train = True, dropout = 0):
        super(TotalModel, self).__init__()

        self.videoencoder = VMo(is_train)
        self.audioencoder = AMo(dropout=dropout)
        self.index = Weight()
        self.is_train = is_train

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        # i decide to transfer the tensor into size:[batch_size, 1024]

    def projector(self, tensor):
        tensor = tensor.view(tensor.size(0), -1)
        linear = nn.Linear(tensor.size(1), 2048)
        tensor = linear(tensor)
        tensor = F.relu(tensor)

        return tensor

    def forward(self, mels, supplement, video):

        a = self.audioencoder(mels, supplement)

        if self.is_train == True :
            v_g, v_l = self.videoencoder(video)
            v,a = self.index(v_l, a, hp.voutputsize, hp.indexingsize, hp.feature_masks) # 还没写，记得在hp里写

            v_g = self.projector(v_g)
            v_l = self.projector(v_l)
            a = self.projector(a)

            return v_g, v_l, a
        elif self.is_train == False :
            v_l = self.videoencoder(video)
            v, a = self.index(v_l, a, hp.voutputsize, hp.indexingsize, hp.feature_masks)  # 还没写，记得在hp里写

            v = self.projector(v)
            a = self.projector(a)

            return v, a
