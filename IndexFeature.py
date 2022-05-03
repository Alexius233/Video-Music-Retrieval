import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        model.append(nn.BatchNorm2d(output_channels))
    if (Relu):
        model.append(nn.ReLU())
    return nn.Sequential(*model)

def mlp(feature, output_size):
    linear = nn.Linear(feature.size(1), output_size)
    feature = linear(feature)
    feature = F.relu(feature)

    return feature

class Weight(torch.nn.Module):

    def name(self):
        return'AudioWeight'


    def __init__(self, networks=None):
        super(Weight, self).__init__()

        self.key_query_dim = 512
        self.key_conv1x1 = create_conv(self.rnn_input_size, self.key_query_dim)
        self.key_conv1x1.apply(weights_init)

    def forward(self, Visionfeatures, Audiofeatures, voutputsize, feature_masks):  # feature_masks可能是某个超参数

        vsize = Visionfeatures.size()
        Visionfeatures = Visionfeatures.view(vsize[0], vsize[1], -1)
        Visionfeature = mlp(Visionfeatures, Visionfeatures.size(2))

        features = torch.cat((Visionfeature, Audiofeatures), dim=1)
        indexingsize = Visionfeatures.size(2)+Audiofeatures(2)
        features = mlp(features, indexingsize)

        self.feature_keys = self.key_conv1x1(features.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1)
        #传入上一个


        # get  scores by dot product attention

        scores = torch.bmm(features.unsqueeze(1), self.feature_keys).squeeze() / np.sqrt(self.key_query_dim)
        scores = scores * feature_masks + (1 - feature_masks) * -(1e35)  # assign a very small value for padded positions
        normalized_query_scores = self.softmax(scores)
        # weighted sum of feature banks to generate c_next
        #audio_weight = torch.bmm(feature_banks.permute(0, 2, 1), normalized_query_scores.unsqueeze(-1)).squeeze(-1)
        Indexed_feature_v = torch.bmm(Visionfeatures, normalized_query_scores[0:Visionfeatures.size()])
        Indexed_feature_a = torch.bmm(Audiofeatures, normalized_query_scores[Audiofeatures.size():])

        return Indexed_feature_v , Indexed_feature_a