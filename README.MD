# Video Musical Retrieval

A PyTorch implementation of our project, hope it appear in our first paper
Haven't finished yet, maybe it can run in a strange way : )

## File structure

- `Hyperparameters.py` --- hyperparameters
- `AudioDataReader.py` --- read and process the audio data(need to finish)
- `AudioModel.py` --- network uesd wo calculate audio data
- `GQDL.py` --- part of the AudiModel, a multihead attention
- `Loss_function.py` --- you guess what it used to do : ))))))
- `Process2frames.py` --- split the video to frames
- `TCN.py` --- part of the WFN, mainly about Dilated Convolutions
- `TSM_UResNet.py` --- a ResNet with upsampling module and TSM
- `TotalModel.py` --- integrate AudioModel and VideoModel
- `VideoModel.py` --- read and process video data
- `VMR_Dataset.py` --- Dataset
- `TrainModel.py` --- to train
- `WFN.py` --- part of VideoModel
- `videotransforms.py` --- data augumentation for frames

## Dataset

we collect from "tiktok.com" and "douyin.com"
if you need, you can contact with me by sending E-mail : 2071035929@qq.com
