import torch
import torch.nn as nn
from torch.autograd import Variable
from clip import clip
from deepSim import deepSim
import math

#An implementation of paper "It’s Time for Artistic Correspondence in Music and Video_CVPR2022"

# bs = 8
# video = torch.rand(bs, 10, 3, 224, 224)
# music = torch.rand(bs, 10, 129, 128)
# model = total_Model(512, 256)     #512：clip的输出维度  256：deepSim的输出维度
# v, m = model(video,music)
# print(v.shape)    #torch.Size([8, 1024])
# print(m.shape)    #torch.Size([8, 512])


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000):
        """
        :param d_model: pe编码维度，一般与word embedding相同，方便相加
        :param dropout: dorp out
        :param max_len: 语料库中最长句子的长度，即word embedding中的L
        """
        super(PositionalEncoding, self).__init__()
        # 定义drop out
        self.dropout = nn.Dropout(p=dropout)
        # 计算pe编码
        pe = torch.zeros(max_len, d_model) # 建立空表，每行代表一个词的位置，每列代表一个编码位
        position = torch.arange(0, max_len).unsqueeze(1) # 建个arrange表示词的位置以便公式计算，size=(max_len,1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *    # 计算公式中10000**（2i/d_model)
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 计算偶数维度的pe值
        pe[:, 1::2] = torch.cos(position * div_term)  # 计算奇数维度的pe值
        pe = pe.unsqueeze(0)  # size=(1, L, d_model)，为了后续与word_embedding相加,意为batch维度下的操作相同
        self.register_buffer('pe', pe)  # pe值是不参加训练的

    def forward(self, x):
        # 输入的最终编码 = word_embedding + positional_embedding
        x = x + Variable(self.pe[:, :x.size(1)],requires_grad=False) #size = [batch, L, d_model]
        # x = x + self.pe[:, :x.size(1)] #size = [batch, L, d_model]
        return self.dropout(x) # size = [batch, L, d_model]


class TransformerModel(nn.Module):

    def __init__(self,num_encoderLayer,num_decoderLayer,num_encoderHead,num_decoderHead,d_model_encoder,d_model_decoder):
        super(TransformerModel, self).__init__()
        self.encoderLayer = nn.TransformerEncoderLayer(d_model=d_model_encoder,nhead=num_encoderHead)
        self.decoderLayer = nn.TransformerDecoderLayer(d_model=d_model_decoder,nhead=num_decoderHead)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoderLayer,num_layers=num_encoderLayer)
        self.decoder = nn.TransformerDecoder(decoder_layer=self.decoderLayer,num_layers=num_decoderLayer)

    def forward(self, src, tgt):
        mem = self.encoder(src)#encoder的输出作为decoder的输入
        output = self.decoder(tgt, mem)
        return output


class total_Model(nn.Module):

    def __init__(self, video_d_model, music_d_model):
        super(total_Model, self).__init__()
        self.video_list = []
        self.music_list = []
        #TODO:转到GPU上运算
        self.video_model, self.preprocess = clip.load("ViT-B/32", device="cpu")
        self.music_model = deepSim.DeepSim()
        self.video_embedding = PositionalEncoding(d_model=video_d_model, dropout=0.1)
        self.music_embedding = PositionalEncoding(d_model=music_d_model, dropout=0.1)
        #TODO：自行确定encoder/decoder层数以及注意力的head数
        self.video_transformer = TransformerModel(2,2,2,2,video_d_model,video_d_model)
        self.music_transformer = TransformerModel(2,2,2,2,music_d_model,music_d_model)
        self.flatten = nn.Flatten()
        self.video_MLP = nn.Sequential(nn.Linear(in_features=5120, out_features=2048)
                                 ,nn.ReLU()
                                 ,nn.Linear(in_features=2048, out_features=1024))
        self.music_MLP = nn.Sequential(nn.Linear(in_features=2560, out_features=1024)
                                       ,nn.ReLU()
                                       ,nn.Linear(in_features=1024, out_features=512))

    def forward(self, video, music):
        bs = video.shape[0]
        music = torch.unsqueeze(music, 2)  # torch.Size([bs, 10, 1, 129, 128])
        for i in range(bs):
            self.video_list.append(self.video_model.encode_image(video[i]))  # video_list[i]:torch.Size([10, 512])
            self.music_list.append(self.music_model(music[i]))               # music_list[i]:torch.Size([10, 256])
        video_features = torch.stack(self.video_list)
        music_features = torch.stack(self.music_list)
        # print(video_features.shape)  # torch.Size([8, 10, 512])
        # print(music_features.shape)  # torch.Size([8, 10, 256])
        #输入进行positional encoding
        #TODO：decoder的输入还未确定
        src_video = self.video_embedding(video_features)
        src_music = self.music_embedding(music_features)
        #过transformer
        #TODO：每个transformer的第二个src应该改为decoder的输入
        trans_v = self.video_transformer(src_video,src_video)   #第二个src改为decoder输入
        trans_m = self.music_transformer(src_music,src_music)   #第二个src改为decoder输入
        #展平
        temp_v = self.flatten(trans_v)
        temp_m = self.flatten(trans_m)
        #过MLP并返回
        return self.video_MLP(temp_v), self.music_MLP(temp_m)
