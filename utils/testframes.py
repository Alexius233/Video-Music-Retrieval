import torch
import torch.utils.data as data_utl
from videotransforms import Transforms

import numpy as np
import os
import os.path

import cv2
import gc

def testframes(video_file):

    fv = open(video_file,'r')
    
    transforms = Transforms(224)
    
    for line in fv.readlines():
        items = line.strip().split('|')
        image_dir = items[0]
        
        print("  视频地址")
        print(image_dir)
        frames_intensive = []

        count = 0
        
        #print(os.listdir(image_dir))
        numberindex = os.listdir(image_dir)
        lens = len(numberindex)
        for i in range(0, lens):
             
            suffix = numberindex[i]
            #print(suffix)
            img = cv2.imread(os.path.join(image_dir, suffix))[:, :,[2, 1, 0]]
            img =  np.asarray(img).transpose(2,0,1)
            img = torch.from_numpy(img)
            #print(img.shape)
            img = transforms(img)
            
            if i == 0:
                video = img
            else :
                video = torch.cat((video, img))
            
            count = count + 1
            
            del img
            if count == 64 :   # 临时方法，效果应该一样
                break 
                
       
        print(video.shape)    
        
        if video.size(0) < 192 :
            print("警告，L数据大小不对")
            #print(video1.size(0))
    
        del frames_intensive,video
        gc.collect()
            

    fv.close() 
    

video_file = '/root/autodl-tmp/VMR_PRO/train/videofilename.txt'

testframes(video_file)