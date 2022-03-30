import os
import numpy as np
import librosa
import librosa.display

# mel-spectrogram parameters
# SR = 12000  # 采样率
# N_FFT = 512
# HOP_LEN = 256
# DURA = 8.129   # 采样长度
# n_mel = 96

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
            src = np.hstack(src, np.zeros(int(DURA * SR) - n_sample)) # hstack 是连接操作
        elif n_sample > n_sample_wanted:  # if too long, cut
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

        # Perform harmonic percussive source separation (HSS)
        y_harmonic, y_percussive = librosa.effects.hpss(src)
        logam = librosa.amplitude_to_db
        MelSpectrogram = []
        fv_total = []

        # for Spectral features
        for i in range(2):
            if i == 0:
                y = y_harmonic
            else:
                y = y_percussive

            fv = logam(librosa.feature.chroma_stft(y=y, sr=SR, hop_length=HOP_LEN, n_fft=N_FFT))  #  -> stft
            if i == 0:
                fv_total = fv
            else:
                fv_total = fv
                fv_total = np.vstack((fv_total, fv))

            fv = logam(librosa.feature.chroma_cens(y=y, sr=SR, hop_length=HOP_LEN), ref_power=np.max)
            fv_total = np.vstack((fv_total, fv))

            fv = logam(librosa.feature.melspectrogram(y=y, hop_length=HOP_LEN, n_fft=N_FFT, n_mels=128),
                       ref_power=np.max)   # stft -> mel
            # log的mel谱
            # hop_len 的作用是 : Sr * DURA / hop_len 影响wide, n_mel 的作用是 : 决定height,n_mel是多少height是多少
            # n_fft : 似乎是影响了分贝数,特征的整体数值大小

            fv_total = np.vstack((fv_total, fv))
            MelSpectrogram = fv_total

            fv_mfcc = librosa.feature.mfcc(y=y, sr=SR, hop_length=HOP_LEN)
            fv = logam(fv_mfcc, ref_power=np.max)
            fv_total = np.vstack((fv_total, fv))
            fv = logam(librosa.feature.delta(fv_mfcc), ref_power=np.max)
            fv_total = np.vstack((fv_total, fv))
            fv = logam(librosa.feature.delta(fv_mfcc, order=2), ref_power=np.max)
            fv_total = np.vstack((fv_total, fv))

            fv = logam(librosa.feature.rms(y=y, hop_length=HOP_LEN), ref_power=np.max)
            fv_total = np.vstack((fv_total, fv))

            fv = logam(librosa.feature.spectral_centroid(y=y, sr=SR, hop_length=HOP_LEN, n_fft=N_FFT), ref_power=np.max)
            fv_total = np.vstack((fv_total, fv))

            fv = logam(librosa.feature.spectral_bandwidth(y=y, sr=SR, hop_length=HOP_LEN, n_fft=N_FFT),
                       ref_power=np.max)
            fv_total = np.vstack((fv_total, fv))

            fv = logam(librosa.feature.spectral_rolloff(y=y, sr=SR, hop_length=HOP_LEN, n_fft=N_FFT), ref_power=np.max)
            fv_total = np.vstack((fv_total, fv))

            fv = logam(librosa.feature.poly_features(y=y, sr=SR, hop_length=HOP_LEN, n_fft=N_FFT), ref_power=np.max)
            fv_total = np.vstack((fv_total, fv))

            fv = logam(librosa.feature.poly_features(y=y, sr=SR, hop_length=HOP_LEN, n_fft=N_FFT, order=2),
                       ref_power=np.max)
            fv_total = np.vstack((fv_total, fv))

            fv = logam(librosa.feature.zero_crossing_rate(y=y, hop_length=HOP_LEN, frame_length=N_FFT),
                       ref_power=np.max)
            fv_total = np.vstack((fv_total, fv))


        # Feature aggregation
        fv_mean = np.mean(fv_total, axis=1)
        fv_var = np.var(fv_total, axis=1)
        fv_amax = np.amax(fv_total, axis=1)
        fv_aggregated = np.hstack((fv_mean, fv_var))
        fv_aggregated = np.hstack((fv_aggregated, fv_amax))

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

        if is_train:
            return MelSpectrogram,fv_mean,fv_var,fv_amax

        else:
            return MelSpectrogram,fv_mean,fv_var,fv_amax, audio_name

