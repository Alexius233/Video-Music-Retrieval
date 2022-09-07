class Hyperparameters():

    #log directory
    log_dir = '/root/autodl-tmp/log'


    # mel-spectrogram parameters
    SR = 16000  # 采样率
    N_FFT = 512
    HOP_LEN = 256
    DURA = 8.192   # 采样长度

    """
    输入的mel-spectrogram 规格是 128 * 512
         supplement             89 * 512
    """

    #GPU
    device = 'cuda:0'

    #train parameters
    lr = 0.0005
    batch_size = 16 
    num_epochs = 2
    eval_size = 1
    save_per_epoch = 1
    log_per_step = 1
    seed = 42
    lr_step = [500000, 1000000, 2000000]  # 可以改

    #test parameters
    test_batch_size = 10
    test_per_epoch = 1000

    ## Audio part
    #WFN
    TCN_input_size = 128   # 输入深度
    TCN_output_size = 128  #应该是与上面一致
    num_channels = [128] * 4    # hidden layer 宽度不变
    TCN_kernel_size = 4
    Bottleneck_output_size = 64  # 瓶颈层中的压缩的那层参数, 试试压缩一半
    # 走完 TCN 规格还是 128 * 512
    out2pool = 128   #一个1*1卷积的输出, 先试试不变
    Pooling_outsize = 64  # AdaptiveMaxpool
    # 现在是 128 * 256
    Depth = 9     # WFN模块需要进行的次数

    #Vice_audionet
    supplement_transform_features = 64     
    af_dim = 96  # 尝试为压缩一半
    meltingc = 192   # 128 + 64

    # GQDL
    token_num = 10
    num_heads = 4
    token_emb_size = 128
    # output : 10 * 128 : [batch_size, 128, 256]


    ## Video part
    
    # because of the complexity, i set most parameters in TSM_UResNet file directly
    #ResNet_front
    layers = None  # layers是[ , , , ]的四个参数的Tuple，表示4个部分的数量参数
    num_frames = 8 # not use
    # output size  = [batch_size, 512, 7,7]

    #VMP_dataset
    root1 = '/root/autodl-tmp/VMR_PRO/train'
    root2 = '/root/autodl-tmp/VMR_PRO/valid'
    root3 = '/root/autodl-tmp/VMR_PRO/test'
    start = 1
    strategy1 = 'intensive'
    strategy2 = 'sparse'     # 目前作废

    # MLP
    n_features1 = 512 * 28 * 28   # video
    n_features2 = 128 * 10      # audio
    n_features  = 2048


    # Loss
    margin = 1
    bias = batch_size
    temperature = 10 #100也可以试试
    balance_parameter = 0.8  # 先试试

    # Align
    feature_masks = 0.9