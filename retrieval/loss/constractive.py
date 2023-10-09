import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    """ Contrastive loss with hard positive/negative mining """
    def __init__(self, margin=0.7, eps=1e-6):
        super(ContrastiveLoss, self).__init__()

        self.margin = margin
        self.eps = eps

    def forward(self, x, label):
        # x is D x N
        x = x.permute(1, 0)
        dim = x.size(0)

        # number of tuples
        nq = torch.sum(label.data == -1)

        # Number of images per tuple Nq+ Np+ Nn
        S = torch.div(x.size(1), nq, rounding_mode='floor')

        xp = x[:, ::S].permute(1, 0).repeat(
            1, S-1).view((S-1)*nq,    dim).permute(1, 0)
        idx = [i for i in range(len(label)) if label.data[i] != -1]

        xn = x[:, idx]
        lbl = label[label != -1]

        diff = xp - xn

        D = torch.pow(diff + self.eps, 2).sum(dim=0).sqrt()

        y = 0.5 * lbl * torch.pow(D, 2) + 0.5 * (1 - lbl) * \
            torch.pow(torch.clamp(self.margin - D, min=0), 2)

        y = torch.sum(y)

        return y

    def __repr__(self):
        return f'{self.__class__.__name__}(margin={self.margin:.4})'
