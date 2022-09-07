import os
import numpy as np
import librosa


# mel-spectrogram parameters
# SR = 16000  # 采样率
# N_FFT = 512
# HOP_LEN = 256
# DURA = 8.192   # 采样长度
# n_mel = 128

# hop_len 的作用是 : Sr * DURA / hop_len 影响wide, n_mel 的作用是 : 决定height,n_mel是多少height是多少
# n_fft : 似乎是影响了分贝数,特征的整体数值大小

def Audio_feature_extractionder(indir, SR, N_FFT, HOP_LEN, DURA, is_train = True):

        # Load audio
        audio_name = indir
        src, sr = librosa.load(indir, sr=SR) # value num = Sr * DURA

        # Trim audio
        n_sample = src.shape[0]
        n_sample_wanted = int(DURA * SR)

        time_window = 25  # 时间窗长度
        window_length = sr / 1000 * time_window  # 转换过来的视窗长度
        window_nums = int(n_sample_wanted / window_length)  # 视窗个数

        if n_sample < n_sample_wanted:  # if too short, pad zero
            src = np.hstack((src, np.zeros(int(DURA * SR) - n_sample))) # hstack 是连接操作
        
        elif n_sample > n_sample_wanted:  # if too long, cut 警告：有问题（改完删去）
            """
            stride = int((n_sample_wanted - window_nums * 400) / (window_nums - 1))
            end_index = stride * (window_nums - 1)
            data = []

            for i in range(0, end_index):
                p_start = i
                p_end = i + window_length

                data_line = src[0, p_start:p_end]  # 截取

                # x = np.linspace(0, window_nums - 1, window_nums, dtype=np.int64)
                # w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (window_nums - 1))  # 汉明窗
                # data_line = data_line * w

                data = np.hstack((src, data_line))

            src = data
            """
            coeff = 0.97#预加重系数
            time_window = 25  # 时间窗长度
            window_length = int(sr / 1000 * time_window)  # 转换过来的视窗长度：400
            frameNum = int(n_sample_wanted / window_length)  # 视窗个数:325
            n_sample_wanted = 131072         # 音频总长度
            # 规格：325个400大小的帧，共计长度

            frameData = np.zeros((window_length, frameNum))  # 创建一个空的
            #print(frameData.shape)
            #汉明窗
            hamwin = np.hamming(window_length)

            for i in range(frameNum):
                singleFrame =src[np.arange(i * window_length, min(i * window_length +window_length ,n_sample_wanted))]
                singleFrame = np.append(singleFrame[0], singleFrame[:-1] - coeff*singleFrame[1:])#预加重
                #singleFrame = np.append(singleFrame[0], singleFrame[1:] - coeff*singleFrame[:-1])
                frameData[:len(singleFrame),i] = singleFrame
                frameData[:,i] = hamwin * frameData[:,i]#加窗

    
            frameData = np.transpose(frameData)
            length = frameData.shape[0]*frameData.shape[1]
            frameData = np.reshape(frameData, length)

            src  = np.hstack((frameData, np.zeros(int(DURA * SR) - length)))

        # Perform harmonic percussive source separation (HSS)
        
        #src = np.array(src)   # 转np.array ? 测试
        #print("音频地址")
        #print(indir)
        y_harmonic, y_percussive = librosa.effects.hpss(src)
        logam = librosa.amplitude_to_db
        fv_total = []

        fv_mel = logam(librosa.feature.melspectrogram(y=src, hop_length=HOP_LEN, n_fft=N_FFT, n_mels=128))  # -> mel
        MelSpectrogram = fv_mel
        MelSpectrogram = MelSpectrogram[:,0:512]


        # 优先抽取log的mel谱
        # hop_len 的作用是 : Sr * DURA / hop_len 影响wide, n_mel 的作用是 : 决定height,n_mel是多少height是多少
        # n_fft : 似乎是影响了分贝数,特征的整体数值大小

        # for Spectral features
        # 按顺序抽取 色度频率, 色能量归一化, mfcc, 质心, 频带宽度， roll-off frequency, n阶多项式系数, 过零率
        for i in range(2):
            if i == 0:
                y = y_harmonic
            else:
                y = y_percussive

            fv = logam(librosa.feature.chroma_stft(y=y, sr=SR, hop_length=HOP_LEN, n_fft=N_FFT))  #  -> 色度频率
                                        # 高度：n_chroma = 12
            if i == 0:
                fv_total = fv
            else:
                fv_total = fv
                fv_total = np.vstack((fv_total, fv))

            fv = logam(librosa.feature.chroma_cens(y=y, sr=SR, hop_length=HOP_LEN)) #   -> 色能量归一化
                                        # 高度：n_chroma = 12
            fv_total = np.vstack((fv_total, fv))

            fv_mfcc = librosa.feature.mfcc(y=y, sr=SR, hop_length=HOP_LEN)   # —> mfcc
                        # 高度：12 与n_mfcc无关？
            fv = logam(fv_mfcc)
            fv_total = np.vstack((fv_total, fv))
            fv = logam(librosa.feature.delta(fv_mfcc))
            #fv1 = logam(librosa.feature.delta(fv_mfcc, order=2))   # delta后宽度减半
                        # 高度 : 20
            #fv = np.hstack((fv, fv1))
            fv_total = np.vstack((fv_total, fv))

            fv = logam(librosa.feature.rms(y=y, hop_length=HOP_LEN))   # -> 均方根
                        # 高度 : 1
            fv_total = np.vstack((fv_total, fv))

            fv = logam(librosa.feature.spectral_centroid(y=y, sr=SR, hop_length=HOP_LEN, n_fft=N_FFT))    # -> 质心、
                        # 高度 : 1
            fv_total = np.vstack((fv_total, fv))

            fv = logam(librosa.feature.spectral_bandwidth(y=y, sr=SR, hop_length=HOP_LEN, n_fft=N_FFT))   # -> 频带宽度
                        # 高度 : 1
            fv_total = np.vstack((fv_total, fv))

            fv = logam(librosa.feature.spectral_rolloff(y=y, sr=SR, hop_length=HOP_LEN, n_fft=N_FFT))    # -> 衰减率
                        # 高度 : 1

            fv_total = np.vstack((fv_total, fv))

            fv = logam(librosa.feature.poly_features(y=y, sr=SR, hop_length=HOP_LEN, n_fft=N_FFT))    # -> 拟合n阶多项式的系数
                        # 高度 : 2
            fv_total = np.vstack((fv_total, fv))

            fv = logam(librosa.feature.poly_features(y=y, sr=SR, hop_length=HOP_LEN, n_fft=N_FFT, order=2))
                        # 高度 : 3
            fv_total = np.vstack((fv_total, fv))

            fv = logam(librosa.feature.zero_crossing_rate(y=y, hop_length=HOP_LEN, frame_length=N_FFT))   # -> 过零率
                        # 高度 : 1
            fv_total = np.vstack((fv_total, fv))
            
            fv_total = fv_total[:,0:512]


        # Feature aggregation
        #print("补充特征大小")
        #print(fv_total.shape)
        #fv_mean = np.mean(fv_total, axis=1)
        #print("补充mean大小")
        #print(fv_mean.shape)
        #fv_var = np.var(fv_total, axis=1)
        #print("补充var大小")
        #print(fv_var.shape)
        #fv_amax = np.amax(fv_total, axis=1)
        #print("补充amax大小")
        #print(fv_amax.shape)
        #fv_aggregated = np.vstack((fv_total, fv_mean))
        #fv_aggregated = np.vstack((fv_total, fv_var))
        #fv_aggregated = np.vstack((fv_total, fv_amax))
        fv_aggregated = fv_total
        #print("补充特征综合大小")
       # print(fv_aggregated.shape)
        

        # # for tempo features
        # tempo, beats = librosa.beat.beat_track(y=y_percussive, sr=SR)
        # fv_aggregated = np.hstack((fv_aggregated, tempo))

        # # for Rhythm features
        # oenv = librosa.onset.onset_strength(y=y_percussive, sr=SR)
        # tempo = librosa.feature.tempogram(onset_envelope=oenv, sr=SR)
        # tempo_mean = np.mean(tempo, axis=1)
        # tempo_var = np.var(tempo, axis=1)
        # tempo_amax = np.amax(tempo, axis=1)
        # tempo_aggregated = np.hstack((tempo_mean, tempo_var))
        # tempo_aggregated = np.hstack((tempo_aggregated, tempo_amax))

        # for pickle
        """
        pklName = outdir + "/" + audio_name + '.pkl'
        f = open(pklName, 'wb')
        pickle.dump(fv_aggregated, f)
        f.close()
        """

        
        

        if is_train == True:
            #return MelSpectrogram,fv_aggregated
            return np.asarray(MelSpectrogram, dtype=np.float32), np.asarray(fv_aggregated, dtype=np.float32)

        else:
            return MelSpectrogram,fv_aggregated, audio_name

