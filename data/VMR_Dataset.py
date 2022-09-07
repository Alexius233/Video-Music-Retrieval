import torch
import torch.utils.data as data_utl

import numpy as np
import os
import os.path

import cv2

from utils.AudioDataReader import Audio_feature_extractionder as AFE
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


def load_rgb_frames(image_dir, fps, strategy, transforms, is_train = True):

    

    frames_sparse = []
    frames_intensive = []
    video_name = [image_dir]

    fps = int(fps)  # 读fps

    count = 0

    # intensive: 均匀抽64张 , 这个有问题 可能是改了忘记保存了
    if strategy == 'intensive':
        for i in os.listdir(image_dir):
            img = cv2.imread(os.path.join(image_dir, i))[:, :,[2, 1, 0]]  # 可以更新数据增强方式
            img = img.transpose(2,0,1)
            img = torch.from_numpy(img)
            img = img.float()
            img = transforms(img)
            
            if count == 0:
                video1 = img
                video2 = img
            if count % 8 == 0 and count != 0:
                video2 = torch.cat((video2, img))
            if count % 1 == 0 and count != 0:
                video1 = torch.cat((video1, img))
            
            count = count + 1
            
            if count == 64 :   # 临时方法，效果应该一样
                break 
    
    #frames_sparse = frames_intensive[::8]
    """
    if is_train == True:
        return np.asarray(frames_intensive, dtype=np.float32), np.asarray(frames_sparse, dtype=np.float32)

    else:
        return np.asarray(frames_intensive, dtype=np.float32), video_name
    """
    if is_train == True:
        return video1, video2

    else:
        return video1, video, video_name

def get_types_data(root, row):   # root: 数据集地址 ; types: train, test, valid

    #now = 'is row{} now'.format(row)
    #print(now)
    video_file = os.path.join(root, "videofilename.txt")
    audio_file = os.path.join(root, "audiofilename.txt")
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
    fa.close()

    return video_frames_path[row],fps[row],audio_path[row]



class VMR_Dataset(data_utl.Dataset):

    def __init__(self,root, start, strategy, transforms, is_train = True, row=slice(0, None)):

        self.root = root
        self.start = start
        self.strategy = strategy
        self.transforms = transforms
        self.is_train = is_train
        self.AFE = AFE


        videopath,fps,audiopath = get_types_data(self.root,row)
        print('Finish loading data')

        self.videopath = videopath
        self.audiopath = audiopath
        self.fps = fps

    def __getitem__(self, index):

        if self.is_train == True:
            out = {}
            video1, video2 = load_rgb_frames(self.videopath[index], self.fps[index], self.strategy, self.transforms, is_train=self.is_train)
            
            if video1.size(0) < 192 :
                print("警告，L数据大小不对")
                print(video1.size(0))
                print(self.videopath[index])
            if video2.size(0) < 24 :
                print("警告，G数据大小不对")
                print(video2.size(0))
                print(self.videopath[index])
            
           

            spe, supplement = self.AFE(self.audiopath[index], hp.SR, hp.N_FFT, hp.HOP_LEN, hp.DURA)
            spe = torch.from_numpy(spe)
            supplement = torch.from_numpy(supplement)
            #fv_feature = []
            #fv_feature.append(fv_mean)
            #fv_feature.append(fv_var)
            #fv_feature.append(fv_amax)
            #fv_feature = np.asarray(fv_feature, dtype=np.float32)
            #fv_feature = torch.from_numpy(fv_feature)

            #fv_mean = torch.from_numpy(fv_mean)
            #fv_var = torch.from_numpy(fv_var)
            #fv_amax = torch.from_numpy(fv_amax)

           
            
            # return 这么多没问题， dataloder传上去之后还可以拆分
            # 这是起的别名，方便引用

            return {'video1':video1 , 'video2': video2, 'mel': spe, 'fv_feature': supplement}

        else:
            out = {}
            imgs , video_name= load_rgb_frames(self.videopath[index], self.fps[index], self.strategy, is_train=self.is_train)

            spe, supplement, audio_name = self.AFE(self.audiopath[index], hp.SR, hp.N_FFT, hp.HOP_LEN, hp.DURA)
            spe = torch.from_numpy(spe)
            supplement = torch.from_numpy(supplement)
          
            #fv_feature = []
            #fv_feature.append(fv_mean)
            #fv_feature.append(fv_var)
            #fv_feature.append(fv_amax)
            #fv_feature = np.asarray(fv_feature, dtype=np.float32)
            # it seems that it is a redundant code, but i'm too tired to fix
            #out['video'] = self.transforms(imgs)
            #out['mel'] = spe
            #out['fv_feature'] = fv_feature
            #out['video_name'] = video_name
            #out['audio_name'] = audio_name

            fv_feature = torch.from_numpy(fv_feature)
            # it seems that it is a redundant code, but i'm too tired to fix

            video = self.transforms(imgs)

            return {'video':video , 'mel': spe, 'fv_feature': supplement, 'video_name': video_name, 'audio_name': audio_name}



    def __len__(self):
        return len(self.videopath)

"""
def collate_graph_fn(data):
  outs = {}
  for key in ['names', 'attn_fts', 'attn_lens', 'sent_ids', 'sent_lens',
              'verb_masks', 'noun_masks', 'node_roles', 'rel_edges']:
    if key in data[0]:
      outs[key] = [x[key] for x in data]

  batch_size = len(data)

  # reduce attn_lens
  if 'attn_fts' in outs:
    max_len = np.max(outs['attn_lens'])
    outs['attn_fts'] = np.stack(outs['attn_fts'], 0)[:, :max_len]

  # reduce caption_ids lens
  if 'sent_lens' in outs:
    max_cap_len = np.max(outs['sent_lens'])
    outs['sent_ids'] = np.array(outs['sent_ids'])[:, :max_cap_len]
    outs['verb_masks'] = np.array(outs['verb_masks'])[:, :, :max_cap_len]
    outs['noun_masks'] = np.array(outs['noun_masks'])[:, :, :max_cap_len]

    return outs
"""