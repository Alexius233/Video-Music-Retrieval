import torch
import torch.nn as nn
import torch.nn.functional as F
from Hyperparameters import  Hyperparameters as hp

"""
class GQDL(nn.Module):  # Global Quantitative Details Layers

    def __init__(self):  # out2next， 上面来的，到256了
        super().__init__()

        self.embed = nn.Parameter(torch.FloatTensor(hp.batch_size, hp.token_num, hp.token_emb_size // hp.num_heads))
        nn.init.normal_(self.embed, mean=0, std=0.5)
        
        #d_q = token_emb_size = 
        #d_k = token_emb_size // num_heads
        
        self.attention = nn.MultiheadAttention(embed_dim=hp.token_emb_size, num_heads=hp.num_heads) # 可能有问题，需要检查

    def forward(self, inputs):
        query = inputs  # [batch, channels, w = 256]
        keys = F.tanh(self.embed)  # [batch_size, token_num, 256 // num_heads] 做个归一
        value = keys

        Quantitative_Details = self.attention(query, keys, value, hp.num_heads)

        return Quantitative_Details
"""

class GQDL(nn.Module):

    def __init__(self):

        super().__init__()

        self.stl = STL()

    def forward(self, inputs):

        style_embed = self.stl(input)

        return style_embed

class STL(nn.Module):
    '''
    inputs --- [N, E//2]
    '''

    def __init__(self):

        super().__init__()
        self.embed = nn.Parameter(torch.FloatTensor(hp.token_num, hp.token_emb_size // hp.num_heads))
        d_q = hp.token_emb_size // 2
        d_k = hp.token_emb_size // hp.num_heads    # 这两个写着意义不明？ 进去linear维度变成了num_unit,也许是增强效果
        self.size = [256, 128]
        self.position = LearnedPositionEncoding(self.size)
        #self.attention = MultiHeadAttention(query_dim=d_q, key_dim=d_k, num_units=hp.token_emb_size, num_heads=hp.num_heads)
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=hp.num_heads,vdim=d_k)

        nn.init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, inputs):
        # Q K V 初始化可能有问题
        N = inputs.size(0)
        query = inputs.unsqueeze(1)  # [N, 1, E//2]
        keys = F.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, E // num_heads]
        value = F.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)
        Quantitative_Details = self.attention(query, keys, value)

        return Quantitative_Details


# 不可学习的位置编码
class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
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
    def __init__(self, embedding_num, dropout=0, max_len=1000):
        super().__init__(max_len, embedding_num)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        weight = self.weight.data.unsqueeze(1)
        x = x + weight[:x.size(1)]
        return self.dropout(x)

