import torch
import torch.nn as nn
from Hyperparameters import Hyperparameters as hp



def Ecos_sim(V_F, A_F, margin=0):  # vision feature, audio feature
    # vision = tensor[batch_size, channels, Height, Wide] -> tensor[channels, Height, Wide]
    # audio  = tensor[batch_szie, channels, Wide]         -> tensor[channels, Wide]
    sizeV = V_F.size()
    sizeA = A_F.size()

    V_F = V_F.view(sizeV[0] * sizeV[1], sizeV[2])
    A_F = A_F.view(sizeA[0] * sizeA[0])

    upper = torch.mm(V_F, A_F)

    V_F_norm = torch.sqrt((V_F ** 2).sum(1).view(-1, 1) + 1e-18)
    A_F_norm = torch.sqrt((A_F ** 2).sum(0) + 1e-18)

    sim = upper / (V_F_norm * A_F_norm)

    sim = torch.exp(sim - margin)

    return sim


class ContrastiveLoss(nn.Module):
    def __init__(self,num=32):  # 假设默认num=32是batch_size
        super(ContrastiveLoss, self).__init__()
        self.num = num


    def forward(self, pre_VF, pre_AF, back_VF, back_AF):
        sim_pre = []
        sim_back_V = []
        sim_back_V_down = []
        sim_back_A = []
        sim_back_A_down = []
        # size = batch_size
        size = self.pre_VF.size1(0)

        for i in range(0, size):
            sim_pre += torch.log(Ecos_sim(pre_VF[:i, ], pre_AF[:i, ]))

        for j in range(0, size):  # vision to audio
            # now = back_VF[:j,]

            if self.num == 32:
                for k in range(0, size):
                    for j in range(0, size):
                        if k == j:
                            continue
                        negtive = Ecos_sim(back_VF[:k, ], back_AF[:j, ])
                        sim_back_V_down += negtive

                    positive = Ecos_sim(back_VF[:k, ], back_AF[:k, ], margin=hp.margin)

                    sim_back_V += torch.log(positive / positive + sim_back_V_down)
            else:
                for k in range(0, size):  # 从这个之后取，到末尾了还没取到对应数量，从头接着取
                    for j in range(k, k + size):
                        negtive = Ecos_sim(back_VF[:k, ], back_AF[:j, ])
                        sim_back_V_down += negtive
                        if j == size:
                            breakpoint = j
                            for j in range(0, size - breakpoint):
                                sim_back_V_down += negtive

                    positive = Ecos_sim(back_VF[:k, ], back_AF[:k, ], margin=self.margin)

                    sim_back_V += torch.log(positive / positive + sim_back_V_down)

        for j in range(0, size):  # audio to vision
            # now = back_VF[:j,]

            if self.num == 32:
                for k in range(0, size):
                    for j in range(0, size):
                        if k == j:
                            continue
                        negtive = Ecos_sim(back_AF[:k, ], back_VF[:j, ])
                        sim_back_V_down += negtive

                    positive = Ecos_sim(back_AF[:k, ], back_VF[:k, ], margin=hp.margin)

                    sim_back_A += torch.log(positive / positive + sim_back_A_down)
            else:
                for k in range(0, size):  # 从这个之后取，到末尾了还没取到对应数量，从头接着取
                    for j in range(k, k + size):
                        negtive = Ecos_sim(back_AF[:k, ], back_VF[:j, ])
                        sim_back_A_down += negtive
                        if j == size:
                            breakpoint = j
                            for j in range(0, size - breakpoint):
                                sim_back_A_down += negtive

                    positive = Ecos_sim(back_VF[:k, ], back_AF[:k, ], margin=hp.margin)

                    sim_back_A += torch.log(positive / positive + sim_back_A_down)


        return hp.balance_parameter * (- 1 / hp.bias * (sim_back_V + sim_back_A)) + (1 - hp.balance_parameter)* sim_pre