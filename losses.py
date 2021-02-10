import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * torch.pow((1-pt),self.gamma) * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class ArcFaceLossAdaptiveMargin(nn.Module):
    def __init__(self, margins=0.5, s=64.0):
        super().__init__()
        # self.crit = DenseCrossEntropy()
        self.s = s
        self.margins = margins

        self.cos_m = math.cos(margins)
        self.sin_m = math.sin(margins)
        self.th = math.cos(math.pi - margins)
        self.mm = math.sin(math.pi - margins) * margins
            
    def forward(self, logits, labels):
        # ms = []
        # ms = self.margins[labels.cpu().numpy()]
        # cos_m = torch.from_numpy(np.cos(ms)).float().cuda()
        # sin_m = torch.from_numpy(np.sin(ms)).float().cuda()
        # th = torch.from_numpy(np.cos(math.pi - ms)).float().cuda()
        # mm = torch.from_numpy(np.sin(math.pi - ms) * ms).float().cuda()
        # labels = F.one_hot(labels, out_dim).float()

        logits = logits.float()
        cosine = logits

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = F.binary_cross_entropy_with_logits(output, labels)
        return loss
