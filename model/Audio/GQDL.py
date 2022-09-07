import torch
import torch.nn as nn
import torch.nn.functional as F
from Hyperparameters import  Hyperparameters as hp



class GQDL(nn.Module):

    def __init__(self):

        super().__init__()

        self.stl = STL()

    def forward(self, inputs):

        style_embed, weight = self.stl(inputs)

        return style_embed

class STL(nn.Module):

    def __init__(self):

        super().__init__()
        
        #self.embed = nn.Parameter(torch.FloatTensor(256， 128))
        self.embed = nn.Parameter(torch.FloatTensor(256,10))#.cuda()
        
        self.position = LearnedPositionEncoding(64).cuda()
        
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, kdim=10,vdim=10, batch_first=True)

        nn.init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, inputs):
        # Q K V 初始化可能有问题
        N = inputs.size(0)
        #query = inputs.unsqueeze(1)  # [N, 1, E//2]
        inputs = self.position(inputs)
       # print(self.position.shape)
        query = inputs.transpose(1,2) 
        #query = inputs
        #print(query.shape)
        
        #keys = torch.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, E ]
        #keys = query
        keys = torch.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)
        #print(keys.shape)
        
        value = torch.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)
        #print(value.shape)
        
        Q, weight = self.attention(query, keys, value)

        return Q, weight


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout=0, max_len=256):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建⼀个⾜够⻓的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
                        -1, 1) / torch.pow(10000, torch.arange(
                         0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)
        
    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

# 可学习的
class LearnedPositionEncoding(nn.Embedding):
    def __init__(self, embedding_num, dropout=0, max_len=256):
        super().__init__(max_len, embedding_num)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        weight = self.weight.data.unsqueeze(1)
        #print(weight.shape)
        #print(x.shape)
        x = x + weight[:x.size(0),:]
        #print(weight[:x.size(0),:].shape)
        return self.dropout(x)


