import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

import numpy as np
import csv

import os
import os.path

import cv2

from AudioDataReader import Audio_feature_extractionder as AFE
from Hyperparameters import Hyperparameters as hp

# attention: 这里处理完输入长度是不固定的

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)

    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))


def load_rgb_frames(image_dir, fps, strategy, is_train = True, start = 1):

    frames = []
    video_name = [image_dir]

    fps = int(fps)  # 读fps

    count = 0
    # 策略待定，随便写的
    # intensive: 一个时间段连续取好几个, 3 + 3 + 3
    if strategy == 'intensive':
        for i in sorted(os.listdir(image_dir)):
            count = count + 1
            for j in range(count, count + int(1.5 * fps)):
                img = cv2.imread(os.path.join(image_dir, str(i).zfill(6) + '.jpg'))[:, :,[2, 1, 0]]  # 某种转置，方便数据后续转成需要的格式
                w, h, c = img.shape
                if w < 226 or h < 226:
                    d = 226. - min(w, h)
                    sc = 1 + d / min(w, h)
                    img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
                img = (img / 255.) * 2 - 1
                j = j + int(0.5 * fps)
                frames.append(img)

    # sparse： 均匀取, 8张
    if strategy == 'sparse':
        size = len(os.listdir(image_dir)) - 0.5 * fps
        gap = size / 8
        count = 0
        number = 0.25 * fps
        while count <= 9:
            if count == 9:
                count = 0
                number = 0.25 * fps
                break

            img = cv2.imread(os.path.join(image_dir, str(number).zfill(6) + '.jpg'))[:, :, [2, 1, 0]]  # 某种转置，方便数据后续转成需要的格式
            w, h, c = img.shape
            if w < 226 or h < 226:
                d = 226. - min(w, h)
                sc = 1 + d / min(w, h)
                img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
            img = (img / 255.) * 2 - 1
            frames.append(img)

            count += 1
            number += gap


    if is_train:
        return np.asarray(frames, dtype=np.float32)

    else:
        return np.asarray(frames, dtype=np.float32), video_name

def get_types_data(root, types, row):   # root: 数据集地址 ; types: train, test, valid

    video_file = os.path.join(os.path.join(root, types), "videofilename.txt")
    audio_file = os.path.join(os.path.join(root, types), "audiofilename.txt")
    video_frames_path = []
    audio_path = []
    fps = []

    fv = open(video_file,'r')
    for line in fv.readlines():
        items = line.strip().split('|')
        video_frames_path.append(items[0])
        fps.append((items[1]))
    fv.close()

    fa = open(audio_file,'r')
    for line in fa.readlines():
        item = line.strip()
        audio_path.append(item)

    return video_frames_path[row],fps[row],audio_path[row]



class VMR_Dataset(data_utl.Dataset):

    def __init__(self,root, start, strategy, is_train = True, transforms=None, row=slice(0, None)):

        #self.data = load_rgb_frames(image_dir, vid, start, num)

        self.start = start
        self.strategy = strategy
        self.transforms = transforms
        self.is_train = is_train
        self.AFE = AFE


        videopath,fps,audiopath = get_types_data(root, row)

        self.videopath = videopath
        self.audiopath = audiopath
        self.fps = fps

    def __getitem__(self, index):

        if self.is_train:
            out = {}
            imgs = load_rgb_frames(self.videopath[index], self.fps[index], self.strategy, is_train=self.is_train)

            imgs = video_to_tensor(imgs)
            # 还需要写一个 压缩长度的处理函数

            spe, fv_mean,fv_var,fv_amax= self.AFE(self.audiopath[index], hp.SR, hp.N_FFT, hp.HOP_LEN, hp.DURA)
            spe = torch.from_numpy(spe)
            fv_feature = []
            fv_feature.append(fv_mean)
            fv_feature.append(fv_var)
            fv_feature.append(fv_amax)
            fv_feature = np.asarray(fv_feature, dtype=np.float32)

            #fv_mean = torch.from_numpy(fv_mean)
            #fv_var = torch.from_numpy(fv_var)
            #fv_amax = torch.from_numpy(fv_amax)

            out['video'] = imgs
            out['mel']   = spe
            out['fv_feature'] = fv_feature

            # return 这么多没问题， dataloder传上去之后还可以拆分
            # 这是起的别名，方便引用

            return out

        else:
            out = {}
            imgs , video_name= load_rgb_frames(self.videopath[index], self.fps[index], self.strategy, is_train=self.is_train)

            spe, fv_mean,fv_var,fv_amax, audio_name= self.AFE(self.audiopath[index], hp.SR, hp.N_FFT, hp.HOP_LEN, hp.DURA)
            spe = torch.from_numpy(spe)
            fv_feature = []
            fv_feature.append(fv_mean)
            fv_feature.append(fv_var)
            fv_feature.append(fv_amax)
            fv_feature = np.asarray(fv_feature, dtype=np.float32)
            spe = torch.from_numpy(spe)
            fv_feature = []
            fv_feature.append(fv_mean)
            fv_feature.append(fv_var)
            fv_feature.append(fv_amax)
            fv_feature = np.asarray(fv_feature, dtype=np.float32)
            # it seems that it is a redundant code, but i'm too tired to fix
            out['video'] = imgs
            out['mel'] = spe
            out['fv_feature'] = fv_feature
            out['video_name'] = video_name
            out['audio_name'] = audio_name

            return out


    def __len__(self):
        return len(self.videopath)



