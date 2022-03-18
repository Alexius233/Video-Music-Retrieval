import numpy as np
import torch

"""
Notation:  dataset里要写新的传递inference数据的接口

"""


def cosine_sim(im, s):          # 已修改，无问题
    '''cosine similarity between all the image and sentence pairs
    '''
    inner_prod = im.mm(s.t())   # .mm() 矩阵乘法 ， .t()二维矩阵转置
    im_norm = torch.sqrt((im**2).sum(1).view(-1, 1) + 1e-18)  # margin = 1e-18
    s_norm = torch.sqrt((s**2).sum(1).view(1, -1) + 1e-18)
    sim = inner_prod / (im_norm * s_norm)

    return sim


def generate_scores(self, **kwargs):     # 生成分数的核心就是 cos相似度   # 已修改，无问题
    # compute image-sentence similarity
    vid_embeds = kwargs['vid_embeds']
    cap_embeds = kwargs['cap_embeds']
    scores = cosine_sim(vid_embeds, cap_embeds) # s[i, j] i: im_idx, j: s_idx
    return scores


def evaluate_scores(tst_reader):  # tst_reader是个dataloader ，       评估的是一个视频对一个批次音频的相似度   ， 未修改

    video_names, all_scores = [], []   #  video名字, 分数 的 数组
    audio_names = tst_reader.dataset.captions  # 不一样，需要改

    for video_data in tst_reader:   # 这么读取是dataset写好的固定方式，我还没写
        video_names.extend(video_data['names'])
        video_enc_outs = forward_video_embed(video_data)
        all_scores.append([])

        for audio_data in tst_reader.dataset.iterate_over_captions(self.config.tst_batch_size):
            audio_enc_outs = forward_audio_embed(audio_data)
            audio_enc_outs.update(video_enc_outs)
            scores = generate_scores(**audio_enc_outs)  # video 对 audio 的score
            all_scores[-1].append(scores.data.cpu().numpy())    # 把分数添上去

        all_scores[-1] = np.concatenate(all_scores[-1], axis=1)
    all_scores = np.concatenate(all_scores, axis=0)             # (n_video, n_audio) 二维数组， 每行对应的是不同的video，每列是对应的排好序的audio的分数

    return video_names, audio_names, all_scores



def evaluate(tst_reader):     #  单一计算分数的函数   ，  # 已修改，无问题

    video_names, audio_names, scores = evaluate_scores(tst_reader)  # 接收video, audio, 和对应分数矩阵（应该就是正方形的）

    ranking_list = []        # 创立排名矩阵
    lens = len(scores[0])    # 多少个audios

    for i in video_names:   # videonames遍历(可能存在问题)
        ranking_list.append([])     # 加一行
        while len(scores[i]) != 0 :
            max = 0
            j = 0
            index = 0

            for j in range(0, len(scores[i])):
                if scores[i][j] >= max:
                    index = j
                    max = scores[i][j]

            ranking_list[-1].append(audio_names[index])   # 写上audio名字
    """
    outs = {
    'video_names': video_names,
    'audio_names': audio_names,
    'ranking_': ranking_list,
    }
    return out
    """
    # 你也可以用这个返回，但是我不太会

    return video_names, audio_names, ranking_list




def test(tst_reader, log_dir, load = True):  # 综合的： 读取，计算，写入   # 未修改完全
    if log_dir is not None and load == True:
        load_checkpoint(log_dir)      # 读取这个点的模型存档，写想读的存档点

    eval_start()

    outs = evaluate(tst_reader)

    with open(log_dir, 'wb') as f:  # log_dir要改，写想存的文件
        # 需要写入文档记录一下


    return outs