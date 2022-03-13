import torch
import torch.nn as nn
import torch.nn.functional as F
from Hyperparameters import Hyperparameters as hp
import TCN


class WPN(nn.Module):
    def __init__(self, dropout):
        super(WPN, self).__init__()
        self.WPN_p1 = TCN.TCN(hp.TCN_input_size, hp.TCN_output_size, hp.spec_channels, dropout=dropout, kernel_size=hp.TCN_kernel_size)
        self.WPN_p2 = WPNBlock(hp.TCN_output_size, hp.Bottleneck_output_size, hp.Poolingsize, dropout=dropout)
        self.Maxpool = nn.MaxPool1d(hp.Poolingsize)
        self.k_layers = hp.Depth

    def forward(self, x):
        skip_connections = []

        for i in range(0, self.k_layers):
            out_self = self.WPN_p1(x)
            out_self, skip_temp = self.WPN_p2(out_self)
            skip_connections += skip_temp

        out = self.Maxpool(skip_connections)

        return out


class WPNBlock(nn.Module):   # 瓶颈层
    def __init__(self, n_inputs, n_outputs, n_out2pool, dropout=0.3):
        super(WPNBlock, self).__init__()
        self.Bottleneck1 = nn.Conv1d(n_inputs, n_outputs, stride=1)
        self.Bottleneck2 = nn.Conv1d(n_outputs, n_inputs, stride=1)

        self.nn.LayerNorm()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.net = nn.Sequential(self.Bottleneck1, self.Bottleneck2,
                                 self.nn.LayerNorm(),
                                 self.relu,
                                 self.dropout)

        self.conv1 = nn.Conv1d(n_inputs, n_out2pool, stride=1)

    def forward(self, x):  # 一个接着进去， 一个出来
        y = self.net(x)
        out_self = y + x
        skip = self.conv1(y)

        return out_self, skip
