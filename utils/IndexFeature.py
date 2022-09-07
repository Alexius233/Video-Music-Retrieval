import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from Hyperparameters import  Hyperparameters as hp


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)


def create_conv(input_channels, output_channels, batch_norm=True, Relu=True):
    model = [nn.Conv1d(input_channels, output_channels, kernel_size=1)]
    if (batch_norm):
        model.append(nn.BatchNorm1d(output_channels))
    if (Relu):
        model.append(nn.ReLU())
    return nn.Sequential(*model)

def build_mlp(input_dim, hidden_dims, output_dim=None, use_batchnorm=True, use_relu=True, dropout=0):
    layers = []
    D = input_dim
    if hidden_dims:
        for dim in hidden_dims:
            layers.append(nn.Linear(D, dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(dim))
            if use_relu:
                layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            D = dim
    if output_dim:
        layers.append(nn.Linear(D, output_dim))
    return nn.Sequential(*layers)


class Weight(nn.Module):


    def __init__(self, length = 1):
        super(Weight, self).__init__()

        self.length = length
        self.key_query_dim_a = 128
        self.key_query_dim_v = 2048
        self.row_input_size_a = 64
        self.row_input_size_v = 49
        
        vkwargs = {
            'input_dim': 2048,
            'hidden_dims': (49,),
            'output_dim' : 2048,
            'use_batchnorm': True,
            'use_relu': True,
            'dropout': 0,
        }
        akwargs = {
            'input_dim': 128,
            'hidden_dims': (64,),
            'output_dim' : 128,
            'use_batchnorm': True,
            'use_relu': True,
            'dropout': 0,
        }
        
        self.key_conv1x1_a = create_conv(self.row_input_size_a, self.row_input_size_a)
        self.key_conv1x1_v = create_conv(self.row_input_size_v, self.row_input_size_v)
        
        # 输入应该是[batchsize, w, h] 改变 h
        self.vision_queryfeature_mlp = build_mlp(**vkwargs)
        self.audio_queryfeature_mlp = build_mlp(**akwargs)
        
        self.softmax = torch.nn.Softmax(dim=-1)
        
        #self.key_conv1x1.apply(weights_init)
    
    def indexforword(self,afeature, vfeature, weightA, weightV):
        
        formerA = afeature
        formerV = vfeature

        # get qurey respectively
        aquery = self.audio_queryfeature_mlp(afeature)
        vquery = self.vision_queryfeature_mlp(vfeature)
        
        # get key 
        akeys = self.key_conv1x1_a(afeature)
        vkeys = self.key_conv1x1_v(vfeature)

        #produce weight
        query_scores_a = torch.bmm(akeys.permute(0,2,1), aquery) / np.sqrt(128)
        #[bs, 128, 128]
        query_scores_v = torch.bmm(vkeys.permute(0,2,1), vquery) / np.sqrt(2048)
        #[bs, 2048, 2048]
        # softmax
        query_scores_a = self.softmax(query_scores_a)
        query_scores_v = self.softmax(query_scores_v)
        
        # 生成index后的
        #afeature = torch.bmm(afeature.permute(0,2,1), query_scores_a).permute(0,2,1)
        afeature = torch.bmm(afeature, query_scores_a)
        #[bs, 64, 128]
        #vfeature = torch.bmm(vfeature.permute(0,2,1), query_scores_v).permute(0,2,1)
        vfeature = torch.bmm(vfeature, query_scores_v)
        #[bs, 49, 2048]
        
        # 层间传递可参考gru
        # v : [batchszie, 49(w), 2048(h)]
        # a : [batchszie, 64(w), 128(h)]
        embed_a = nn.Parameter(torch.FloatTensor(hp.batch_size, 64,128)).cuda()
        nn.init.normal_(embed_a, mean=0, std=0.02)
        embed_v = nn.Parameter(torch.FloatTensor(hp.batch_size, 49,2048)).cuda()
        nn.init.normal_(embed_v, mean=0, std=0.02)
        if torch.isnan(embed_a).sum()>0 or torch.isnan(embed_v).sum()>0:
            print("参数生成存在NaN")

        weightA = torch.bmm(embed_a, query_scores_a.permute(0,2,1))
        weightA =  self.softmax(weightA)
        weightV = torch.bmm(embed_v, query_scores_v.permute(0,2,1))
        weightV =  self.softmax(weightV)
        
        # merge bewteen former and now features
        afeature = weightA * formerA + (1 - weightA) * afeature
        vfeature = weightV * formerV + (1 - weightV) * vfeature
        
        
        return afeature,vfeature,weightA,weightV
    
    
    def indexforwordsingle(self,afeature, vfeature):
        # get qurey respectively
        aquery = self.audio_queryfeature_mlp(afeature)
        vquery = self.vision_queryfeature_mlp(vfeature)
        
        # get key 
        akeys = self.key_conv1x1_a(afeature)
        vkeys = self.key_conv1x1_v(vfeature)

        #produce weight
        #print(akeys.permute(0,2,1).shape)
        #print(aquery.shape)
        query_scores_a = torch.bmm(akeys.permute(0,2,1), aquery) / np.sqrt(128)
        #[bs, 128, 128]
        #print(vkeys.permute(0,2,1).shape)
        #print(vquery.shape)
        query_scores_v = torch.bmm(vkeys.permute(0,2,1), vquery) / np.sqrt(2048)
        #[bs, 2048, 2048]
        # softmax
        query_scores_a = self.softmax(query_scores_a)
        #print(query_scores_a.shape)
        query_scores_v = self.softmax(query_scores_v)
        
        # 生成index后的
        #print(afeature.shape)
        #print(query_scores_a.shape)
        afeature = torch.bmm(afeature, query_scores_a)
        #[bs, 64, 128]
        vfeature = torch.bmm(vfeature, query_scores_v)
        #[bs, 49, 2048]
        
        # 层间传递可参考gru
        # v : [batchszie, 49(w), 2048(h)]    
        # a : [batchszie, 256(w), 128(h)]
        embed_a = nn.Parameter(torch.FloatTensor(hp.batch_size, 64,128)).cuda()
        embed_v = nn.Parameter(torch.FloatTensor(hp.batch_size, 49,2048)).cuda()
        
        
        #print(embed_a.shape)
        #print(query_scores_a.permute(0,2,1).shape)
        weightA = torch.bmm(embed_a, query_scores_a.permute(0,2,1))
        weightA =  self.softmax(weightA)
        weightV = torch.bmm(embed_v, query_scores_v.permute(0,2,1))
        weightV =  self.softmax(weightV)
    
        return afeature,vfeature,weightA,weightV
    
    def forward(self, Visionfeatures, Audiofeatures): 
        
        Visionfeatures = Visionfeatures.reshape(hp.batch_size, 2048, 49).permute(0,2,1)
            
        for step in range(self.length):
            if step == 0:
                Audiofeatures,Visionfeatures,weightA,weightV = self.indexforwordsingle(Audiofeatures,Visionfeatures)
            else:
                Audiofeatures,Visionfeatures,weightA,weightV = self.indexforword(Audiofeatures,Visionfeatures,weightA,weightV)
             
        
        return Visionfeatures , Audiofeatures