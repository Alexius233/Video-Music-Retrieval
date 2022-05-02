
import numpy as np
import torch
import torch.nn as nn


import GQDL
class AudioWeght(torch.nn.Module):

    def name(self):
        return'AudioWeght'

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm2d') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.02)

    def create_conv(input_channels, output_channels, kernel, paddings, batch_norm=True, Relu=True, stride=1):
        model = [nn.Conv2d(input_channels, output_channels, kernel, stride=stride, padding=paddings)]
        if (batch_norm):
            model.append(nn.BatchNorm2d(output_channels))
        if (Relu):
            model.append(nn.ReLU())
        return nn.Sequential(*model)

    def __init__(self, net_classifier, args, networks=None):
        super(AudioWeght, self).__init__()
        self.args = args
        self.net_classifier = net_classifier
        self.key_query_dim = 512

        self.key_conv1x1 = self.create_conv(self.rnn_input_size, self.key_query_dim, 1, 0)
        self.key_conv1x1.apply(self.weights_init)

    def forward(self, features, feature_banks, feature_masks, hx, dropmask=None):
        self.feature_keys = self.key_conv1x1(features.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1)
        #传入上一个
        After_Attention = GQDL.MultiHeadAttention

        # get  scores by dot product attention

        scores = torch.bmm(After_Attention.unsqueeze(1), self.feature_keys).squeeze() / np.sqrt(self.key_query_dim)
        scores = scores * feature_masks + (1 - feature_masks) * -(1e35)  # assign a very small value for padded positions
        normalized_query_scores = self.softmax(scores)


        # weighted sum of feature banks to generate c_next
        audio_weight = torch.bmm(feature_banks.permute(0, 2, 1), normalized_query_scores.unsqueeze(-1)).squeeze(-1)

        return audio_weight
