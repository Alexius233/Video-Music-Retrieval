from Hyperparameters import Hyperparameters as hp
from torch.utils.data import DataLoader
from VMR_Dataset import VMR_Dataset
import numpy as np
import torch
import os
import sys
import torch.optim as optim
from torch.nn import DataParallel
from TotalModel import TotalModel
from Loss_Function import ContrastiveLoss, VideoLoss, TotalLoss
from time import time
from videotransforms import Transforms
from ViewGenerator import ContrastiveLearningViewGenerator as CLV
from torch.utils.tensorboard import SummaryWriter
from inference import assess




def train(log_dir, dataset_size, device, writer, start_epoch=0):


    f = open(os.path.join(log_dir, 'log{}.txt'.format(start_epoch)), 'w')

    msg = 'use {}'.format(hp.device)
    print(msg)


    # 整合的最终模型
    model = TotalModel(0.5).cuda()

    if torch.cuda.device_count() > 1:
       model = DataParallel(model)
        # GPU不止一个，并行计算
    if start_epoch != 0:
        model_path = os.path.join(log_dir, 'state', 'epoch{}.pt'.format(start_epoch))
        model.load_state_dict(torch.load(model_path))  # 加载之前保存的参数, load_state_dict: 加载参数, torch_load: 读取文件内的参数，返回参数
        msg = 'Load model of' + model_path
    else:
        msg = 'New model'

    print(msg)
    f.write(msg + '\n')

    optimizer = optim.Adam(model.parameters(), lr=hp.lr)

    if start_epoch != 0:
        opt_path = os.path.join(log_dir, 'state_opt', 'epoch{}.pt'.format(start_epoch))   # 加载优化器的参数
        optimizer.load_state_dict(torch.load(opt_path))
        msg = 'Load optimizer of' + opt_path
    else:
        msg = 'New optimizer'

    print(msg)
    f.write(msg + '\n')

    model = model.to(device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

    sub_critertion1 = VideoLoss()
    sub_critertion2 = ContrastiveLoss()
    criterion = TotalLoss()  # Loss

    # load data
    if dataset_size is None:
        train_dataset = VMR_Dataset(hp.root1,
                                    hp.start,
                                    hp.strategy1,
                                    transforms=CLV(Transforms(224),
                                                   Transforms(96),
                                                   n_views = 2),
                                    row=slice(hp.eval_size, None))
    else:
        train_dataset = VMR_Dataset(hp.root1,
                                    hp.start,
                                    hp.strategy1,
                                    transforms=CLV(Transforms(224),
                                                   Transforms(96),
                                                   n_views = 2),
                                    row=slice(hp.eval_size, hp.eval_size + dataset_size))

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=hp.batch_size,
                              drop_last=True,
                              num_workers=8,
                              shuffle=True)

    num_train_data = len(train_dataset)
    total_step = hp.num_epochs * num_train_data // hp.batch_size
    start_step = start_epoch * num_train_data // hp.batch_size
    step = 0
    global_step = step + start_step
    prev = beg = int(time())

    for epoch in range(start_epoch + 1, hp.num_epochs):  # 正式开始训练

        model.train(True)
        loss_epoch = 0
        for i, batch in enumerate(train_loader):

            step += 1
            global_step += 1

            mels = batch['mel'].to(device)
            frames1 = batch['videos1'].to(device)   # global
            frames2 = batch['videos2'].to(device)   # local
            frames = torch.stack([frames1, frames2], dim=0)
            supplement = batch['fv_feature'].to(device)  # 三个手工特征

            optimizer.zero_grad()

            video_feature_G, video_feature_L, audio_feature= model(mels,supplement, frames)

            vloss = sub_critertion1(video_feature_L, video_feature_G)
            closs = sub_critertion2(video_feature_L, audio_feature)
            loss = criterion(vloss, closs)
            loss_epoch += loss
            loss.backward()

            writer.add_scalar("Loss/train", loss / len(train_loader), step)  # 一个batch的loss的写入

            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)  # clip gradients
            optimizer.step()  # scheduler.step()


            if global_step in hp.lr_step:  # 变更学习率
                optimizer = set_lr(optimizer, global_step, f, writer=writer)

            if (i + 1) % hp.log_per_batch == 0:   # 写时间
                now = int(time())
                use_time = now - prev
                # total_time = hp.num_epoch * (now - beg) * num_train_data // (hp.batch_size * (i + 1) + epoch * num_train_data)
                total_time = total_step * (now - beg) // step
                left_time = total_time - (now - beg)
                left_time_h = left_time // 3600
                left_time_m = left_time // 60 % 60
                msg = 'step: {}/{}, epoch: {}, batch {}, loss: {:.3f}, mel_loss: {:.3f}, mag_loss: {:.3f}, use_time: {}s, left_time: {}h {}m'
                msg = msg.format(global_step, total_step, epoch, i + 1, loss.item(), use_time, left_time_h, left_time_m)

                f.write(msg + '\n')
                print(msg)

                prev = now

            writer.add_scalar("Loss/train_epoch", loss_epoch / len(train_loader), epoch)   # 写入一个epoch的平均loss


        # save model, optimizer
        if epoch % hp.save_per_epoch == 0 and epoch != 0:

            torch.save(model.state_dict(), os.path.join(log_dir, 'state/epoch{}.pt'.format(epoch)))
            torch.save(optimizer.state_dict(), os.path.join(log_dir, 'state_opt/epoch{}.pt'.format(epoch)))
            msg = 'save model, optimizer in epoch{}'.format(epoch)
            f.write(msg + '\n')
            print(msg)

            model.eval()

        # assses the current model, test_per_epoch is equal to save_per_epoch!
        if epoch % hp.test_per_epoch == 0 and epoch != 0:
            result = assess(log_dir, epoch)

    msg = 'Training Finish !!!!'
    f.write(msg + '\n')
    print(msg)

    f.close()



def set_lr(optimizer, step, f, writer):   # 阶梯形 梯度
    if step == 500000:
        msg = 'set lr = 0.0005'
        new_lr = 0.0005
        f.write(msg)
        print(msg)
        writer = writer.add_scalar("learning_rate/lr", new_lr, step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0005

    elif step == 1000000:
        msg = 'set lr = 0.0003'
        f.write(msg)
        new_lr = 0.0003
        print(msg)
        writer = writer.add_scalar("learning_rate/lr", new_lr, step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0003

    elif step == 2000000:
        msg = 'set lr = 0.0001'
        f.write(msg)
        new_lr = 0.0001
        print(msg)
        writer = writer.add_scalar("learning_rate/lr", new_lr, step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    return optimizer





def main(log_dir,dataset_size):

    torch.manual_seed(hp.seed)
    np.random.seed(hp.seed)
    device = torch.device(hp.device)

    # create files
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists(os.path.join(log_dir, 'state')):  # 保存模型参数
        os.mkdir(os.path.join(log_dir, 'state'))
    if not os.path.exists(os.path.join(log_dir, 'state_opt')):  # 保存 优化器 的参数
        os.mkdir(os.path.join(log_dir, 'state_opt'))
    if not os.path.exists(os.path.join(log_dir, 'rank')):
        os.mkdir(os.path.join(log_dir, 'rank'))

    writer = SummaryWriter()

    train(log_dir, dataset_size, device, writer, start_epoch=0)




if __name__ == '__main__':  # 训练时时候用命令行来启动，argv是在命令行输入的三个参数
    argv = sys.argv
    log_number = int(argv[1])
    start_epoch = int(argv[3])
    if argv[2].lower() != 'all':
        dataset_size = int(argv[2])
    else:
        dataset_size = None

    main(log_dir=hp.log_dir, dataset_size=dataset_size)


