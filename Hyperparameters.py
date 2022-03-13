

class Hyperparameters():

    #log directory
    log_dir = None


    # mel-spectrogram parameters
    SR = 12000  # 采样率
    N_FFT = 512
    HOP_LEN = 256
    DURA = 29.12   # 采样长度

    #GPU
    device = 'cuda:0'

    #train paramters
    lr = 0.001
    batch_size = 16
    num_epochs = 100
    eval_size = 1
    save_per_epoch = 10
    log_per_batch = 20
    seed = 42
    lr_step = [500000, 1000000, 2000000]  # 可以改


    #GQDL
    token_num = 10
    num_heads = 8
    token_emb_size = None   #未定

    #WFN
    TCN_input_size = None
    TCN_output_size = None  #应该是与上面一致
    spec_channels = None    # 谱图的通道数，好像是=n_mel
    TCN_kernel_size = None
    Bottleneck_output_size = None # 瓶颈层中的压缩的那层参数
    Poolingsize = None  # 拿去pooling的size
    Depth = None      # WFN模块需要进行的次数

    #ResNet_front
    layers = None  # layers是[ , , , ]的四个参数的Tuple，表示4个部分的数量参数

    # ResNet_back
    up_paramter1 = None
    up_paramter2 = None
    up_paramter3 = None

    #VMP_dataset
    root1 = None
    root2 = None
    root3 = None
    start = 1
    strategy1 = 'intensive'
    strategy2 = 'sparse'

    # MLP
    n_feature = None


    # Loss
    margin = None
    bias = None
    balance_parameter = None