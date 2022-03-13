import torch
import torch.nn as nn
import torch.nn.functional as F
from Hyperparameters import  Hyperparameters as hp


class GQDL(nn.Module):  # Global Quantitative Details Layers

    def __init__(self):  # out2next， 上面来的，到256了
        super().__init__()

        self.embed = nn.Parameter(torch.FloatTensor(hp.batch_size, hp.token_num, hp.token_emb_size // hp.num_heads))
        nn.init.normal_(self.embed, mean=0, std=0.5)
        """
        d_q = token_emb_size = 
        d_k = token_emb_size // num_heads
        """
        self.attention = nn.MultiheadAttention(embed_dim=hp.token_emb_size, num_heads=hp.num_heads) # 可能有问题，需要检查

    def forward(self, inputs):
        query = inputs  # [batch, channels, w = 256]
        keys = F.tanh(self.embed)  # [batch_size, token_num, 256 // num_heads] 做个归一
        value = keys

        Quantitative_Details = self.attention(query, keys, value, hp.num_heads)

        return Quantitative_Details

