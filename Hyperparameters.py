

class Hyperparameters():

    #log directory
    log_dir = None


    # mel-spectrogram parameters
    SR = 16000  # 采样率
    N_FFT = 512
    HOP_LEN = 256
    DURA = 8.129   # 采样长度

    """
    输入的mel-spectrogram 规格是 96 * 512
    """

    #GPU
    device = 'cuda:0'

    #train parameters
    lr = 0.001
    batch_size = 16     # 可能带不起来，可以缩小
    num_epochs = 100
    eval_size = 1
    save_per_epoch = 10
    log_per_batch = 20
    seed = 42
    lr_step = [500000, 1000000, 2000000]  # 可以改

    #test parameters
    test_batch_size = 10
    test_per_epoch = save_per_epoch

    # Audio part

    #WFN
    TCN_input_size = 96   # 输入深度
    TCN_output_size = 96  #应该是与上面一致
    num_channels = [96] * 4    # hidden layer 宽度不变
    TCN_kernel_size = 4
    Bottleneck_output_size = 48  # 瓶颈层中的压缩的那层参数, 试试压缩一半
    # 走完 TCN 规格还是 96 * 512
    out2pool = 96   # 一个1*1卷积的输出, 先试试不变
    Pooling_outsize = 256  # AdaptiveMaxpool
    # 现在是 96 * 256
    Depth = 9     # WFN模块需要进行的次数

    # GQDL
    token_num = 10
    num_heads = 8
    token_emb_size = 256
    # output : 10 * 96


    # Video part

    #ResNet_front
    layers = None  # layers是[ , , , ]的四个参数的Tuple，表示4个部分的数量参数

    # ResNet_back     # 上采样不一定要这么死板，改成和音频能接上就行
    up_paramter1 = [batch_size * 8, 512, 14, 14]
    up_paramter2 = [batch_size * 8, 512, 28, 28]
    up_paramter3 = [batch_size * 8, 512, 56, 56]

    #VMP_dataset
    root1 = None
    root2 = None
    root3 = None
    start = 1
    strategy1 = 'intensive'  # 目前看来用不着
    strategy2 = 'sparse'

    # MLP
    n_feature = None


    # Loss
    margin = None
    bias = None
    balance_parameter = None