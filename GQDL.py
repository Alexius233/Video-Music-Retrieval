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
        d_k = hp.token_emb_size // hp.num_heads
        # self.attention = MultiHeadAttention(hp.num_heads, d_model, d_q, d_v)
        self.attention = MultiHeadAttention(query_dim=d_q, key_dim=d_k, num_units=hp.token_emb_size, num_heads=hp.num_heads)

        nn.init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, inputs):
        N = inputs.size(0)
        query = inputs.unsqueeze(1)  # [N, 1, E//2]
        keys = F.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, E // num_heads]
        Quantitative_Details = self.attention(query, keys)

        return Quantitative_Details


class MultiHeadAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    '''

    def __init__(self, query_dim, key_dim, num_units, num_heads):

        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)
        scores = F.softmax(scores, dim=3)

        # out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return out