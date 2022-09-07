import torch
import torch.nn as nn
import math
import torch.distributed as dist
from Hyperparameters import Hyperparameters as hp

class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out

def Ecos_sim(V_F_single, A_F_single, margin=0):  # vision feature, audio feature
    # vision = tensor[batch_size, channels, Height, Wide] -> tensor[channels, Height, Wide]
    # audio  = tensor[batch_szie, channels, Wide]         -> tensor[channels, Wide]

    #sizeV = V_F.size()
    #sizeA = A_F.size()

    #V_F = V_F.view(sizeV[0] * sizeV[1], sizeV[2])
    #A_F = A_F.view(sizeA[0] * sizeA[0])

    #upper = V_F_single.mm(A_F_single.t())

    upper = (A_F_single * V_F_single).sum()  # 这里有问题

    V_F_norm = torch.sqrt((V_F_single ** 2).sum() + 1e-18)
    A_F_norm = torch.sqrt((A_F_single ** 2).sum() + 1e-18)

    sim = upper / (V_F_norm * A_F_norm)
    #print(sim)
    #e_sim = torch.tensor(math.exp(sim - margin))
    
    #if torch.isinf(e_sim).sum()>0:
        #print("存在无限大")
    #if torch.isnan(e_sim).sum()>0:
        #print("存在NaN")
        

    return sim              # 只是算一个样本的


class ContrastiveLoss(nn.Module):
    def __init__(self,num=hp.batch_size):  # 假设默认num=32是batch_size
        super(ContrastiveLoss, self).__init__()
        self.num = num
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-18)


    def forward(self, back_VF, back_AF):
        size = hp.batch_size

        #######################################################################
        if self.num == hp.batch_size:
            for i in range(0, size):  # vision to audio, 一个i算一个样本的loss
                for j in range(0, size):
                    #if j == i & i != 0:
                        #continue
    
                    negative = Ecos_sim(back_VF[i], back_AF[j])
                    
                    if j == 0:
                        sim_back_V_down = negative
                    if j != i and j != 0:
                        sim_back_V_down += negative

                positive = Ecos_sim(back_VF[i], back_AF[i], margin=hp.margin)

                if i == 0:
                    sim_back_V = torch.log(positive/ (positive + sim_back_V_down))
                else:
                    sim_back_V += torch.log(positive/ (positive + sim_back_V_down)) 

        """
        else:   # 警告：这个暂时不使用，且存在问题
            for i in range(0, size):  # 从这个之后取，到末尾了还没取到对应数量，从头接着取
                for j in range(i, i + size):
                    negtive = Ecos_sim(back_VF[i], back_AF[j])
                    if i == 0:
                        sim_back_V_down = negtive

                    if j == size:
                        breakpoint = j
                        for j in range(1, size - breakpoint):
                            sim_back_V_down += negtive

                positive = Ecos_sim(back_VF[i], back_AF[i], margin=hp.margin)

                if i == 0:
                    sim_back_V = torch.log(positive / positive + sim_back_V_down)
                else:
                    sim_back_V = torch.cat((sim_back_V, torch.log(positive / positive + sim_back_V_down)), 0)
           """
#######################################################################
        if self.num == hp.batch_size:
            for i in range(0, size):  # audio to vision
                # now = back_AF[i]
                if self.num == hp.batch_size:
                    for j in range(0, size):
                        #if j == i & i != 0:
                            #continue
                        
                        negtive = Ecos_sim(back_AF[i], back_VF[j])
                        
                        if j == 0:
                            sim_back_A_down = negtive
                        if j != i and j != 0:
                            sim_back_A_down += negtive

                positive = Ecos_sim(back_AF[i], back_VF[i], margin=hp.margin)

                if i == 0:
                    sim_back_A = torch.log(positive / (positive + sim_back_A_down))
                else:
                    sim_back_A += torch.log(positive / (positive + sim_back_A_down))
        """
        else:
            for i in range(0, size):  # 从这个之后取，到末尾了还没取到对应数量，从头接着取
                for j in range(i, i + size):
                    negtive = Ecos_sim(back_AF[i], back_VF[j])
                    if j == 0:
                        sim_back_A_down = negtive
                    if j == size:
                        breakpoint = j
                        for j in range(1, size - breakpoint):
                                sim_back_A_down += negtive

                positive = Ecos_sim(back_VF[i], back_AF[i], margin=hp.margin)

                if i == 0:
                    sim_back_A = torch.log(positive/ positive + sim_back_A_down)
                else:
                    sim_back_A = torch.cat((sim_back_A, torch.log(positive / positive + sim_back_V_down)), 0)
        """


        return - 1 / hp.bias * (sim_back_V + sim_back_A)

class CosineSimilarity(nn.Module):
 
    def forward(self, tensor_1, tensor_2):
        normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
        normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
        return (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1)



"""    
class VideoLoss(nn.Module):
    def __init__(self):  # 假设默认num=32是batch_size
        super(VideoLoss, self).__init__()
        self.cos_sim = CosineSimilarity()

    def forward(self, feature_L,feature_G):
        G_feature = feature_G
        L_feature = feature_L
        sim = self.cos_sim(G_feature, L_feature).sum()
        #print(sim)
        return sim
"""
class VideoLoss(nn.Module):
    def __init__(self):
        super(VideoLoss, self).__init__()
        self.batch_size = hp.batch_size
        self.temperature = hp.temperature
        self.world_size = 1*1
        #GPU数 * nodes数

        self.mask = self.mask_correlated_samples(self.batch_size, self.world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size * world_size + i] = 0
            mask[batch_size * world_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as         negative examples.
        """
        N = 2 * self.batch_size * self.world_size

        z = torch.cat((z_i, z_j), dim=0)
        if self.world_size > 1:
            z = torch.cat(GatherLayer.apply(z), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

class TotalLoss(nn.Module):
    def __init__(self):
        super(TotalLoss, self).__init__()
        self.V_weight = 0.5#暂定
        self.C_weight = 0.5#暂定

    def forward(self, V, C):
        #print("loss C")
        #print(C)
        #print("loss V")
        #print(V)
        return V * self.V_weight + C * self.C_weight