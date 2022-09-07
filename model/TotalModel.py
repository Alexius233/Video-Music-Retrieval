import torch.nn.functional as F
import torch.nn as nn
import torch
from model.Audio.AudioModel import Audiomodel as AMo
from model.Vision.VideoModel import VideoModel as VMo
from utils.IndexFeature import Weight
from Hyperparameters import Hyperparameters as hp

def projector(inputsize):
        
    layers = []
    layers.append(nn.Linear(inputsize, 2048))
    layers.append(nn.ReLU(inplace=True))
    
    return nn.Sequential(*layers)    
        
    #tensor = tensor.view(tensor.size(0), -1)
    #linear = nn.Linear(tensor.size(1), 2048)
    #tensor = linear(tensor)
    #tensor = F.relu(tensor)



class TotalModel(nn.Module):
    def __init__(self, is_train = True, dropout = 0):
        super(TotalModel, self).__init__()

        self.videoencoder = VMo(is_train)
        self.audioencoder = AMo(dropout=dropout)
        self.index = Weight(length = 2)
        self.is_train = is_train
        

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        # i decide to transfer the tensor into size:[batch_size, 1024]
        self.projector_l = projector(100352)
        self.projector_g = projector(100352)
        self.projector_a = projector(8192)

    def forward(self, mels, supplement, frames1, frames2):

        a = self.audioencoder(mels, supplement)

        if self.is_train == True :
            v_g, v_l = self.videoencoder(frames1, frames2)    
            v,a = self.index(v_l, a)          
            #if torch.isnan(v).sum()>0 or torch.isnan(a).sum()>0:
                #print("index存在NaN")
            
            v_g = v_g.reshape(hp.batch_size, -1)
            v_l = v_l.reshape(hp.batch_size, -1)
            #print(v_l.shape)
            a = a.reshape(hp.batch_size, -1)
        
            
            v_g = self.projector_g(v_g)  
            v_l = self.projector_l(v_l)
            a = self.projector_a(a)
            
            return v_g, v_l, a
        
        else :

            v_l = self.videoencoder(video)
            v, a = self.index(v_l, a) 
            
            v_l = v_l.view(hp.batch_size, -1)
            a = a.view(hp.batch_size, -1)
            
            v = self.projector_l(v)
            a = self.projector_a(a)

            return v, a
