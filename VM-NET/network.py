import torch
import torch.nn as nn
import inception
import numpy as np
from numpy.linalg import eig


#An implementation of paper "Cbvmr content-based video-music retrieval using soft intra-modal structure constraint"


#   输入：torch.Size([bs,64,3,H,W])
#   1.list1 = extract_feature_multi_video(x)     list1[i].shape:[1024]
#   2.res = torch.stack(list1)                   res.shape:[bs,1024]
#   3.输入Video_Model中进行训练

class Music_Model(nn.Module):
    def __init__(self, in_feature_size, dropout=0.1):
        super(Music_Model, self).__init__()
        self.fc1 = nn.Linear(in_feature_size, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.dropout(x)
        x = self.relu(x)
        #L2-normalized
        x = x/torch.norm(x, p=2, dim=1, keepdim=True)
        x[torch.isnan(x)] = 0
        return x


class Video_Model(nn.Module):
    def __init__(self, in_feature_size, dropout=0.1):
        super(Video_Model, self).__init__()
        self.fc1 = nn.Linear(in_feature_size, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)
        #L2-normalized
        x = x/torch.norm(x, p=2, dim=1, keepdim=True)
        x[torch.isnan(x)] = 0
        return x


def pca(X, k, whiten=False):
    X = X - X.mean(axis=0)  # 列向量X去中心化
    X_cov = np.cov(X, ddof=0)  # 计算向量X的协方差矩阵，自由度可以选择0或1
    eigenvalues, eigenvectors = eig(X_cov)  # 计算协方差矩阵的特征值和特征向量（特征向量按列来看）
    # 选取最大的K个特征值及其特征向量
    klarge_index = eigenvalues.argsort()[-k:][::-1]
    k_eigenvalues = eigenvalues[klarge_index]
    k_eigenvectors = eigenvectors[:, klarge_index]
    out = np.dot(X.T, k_eigenvectors).reshape((1024,))  # 用X与特征向量相乘 (1024,)表示一维数组，有1024个元素

    # Whiten
    if whiten:
        out /= np.sqrt(k_eigenvalues + 1e-4)
    return torch.from_numpy(out)


def extract_feature_single_video(x,model):  # x:[64,3,299,299]
    num_of_pic = x.shape[0]
    out_cnn, aux = model(x) # [64,2048,1,1]  aux是inception的另一个返回值，不用管
    list = []
    for i in range(num_of_pic):
        out_pca = pca(out_cnn[i].detach().numpy().reshape(2048, 1), 1024, whiten=True)  # [1024]
        mean = torch.mean(out_pca)
        std = torch.std(out_pca)
        out_pca -= mean
        out_pca /= std
        top_16 = torch.topk(out_pca, 16).values
        list.append(top_16)

    feat = torch.zeros(1024)
    for i in range(num_of_pic):
        if i == 0:
            feat = list[i]
        else:
            feat = torch.cat((feat, list[i]))
    feat -= torch.mean(feat)
    return feat


def extract_feature_multi_video(x):    #x:[bs,64,3,299,299]
    model = inception.inception_v3()
    bs = x.shape[0]
    list = []
    for i in range(bs):
        list.append(extract_feature_single_video(x[i],model))  #list[i]:[1024]，即一个视频的特征
    return list
