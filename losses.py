import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


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

        logits = logits.float()
        cosine = logits

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = F.binary_cross_entropy_with_logits(output, labels)
        return loss


class DeepAUC(nn.Module):
    def __init__(self, phat):
        super(DeepAUC, self).__init__()
        self.phat = phat
        self.margin = 0.5

    def forward(self, mod_out, labels, expt_a, expt_b, alpha):

        logits = torch.sigmoid(mod_out)
        neg_ind = torch.relu(-1*(labels-1))

        A1 = torch.mean((1-self.phat)*torch.pow(logits - expt_a, 2)*labels.float(), dim=0)
        A2 = torch.mean(self.phat*torch.pow(logits - expt_b, 2)*neg_ind.float(), dim=0)
        cross_term = self.phat*(1-self.phat)*(alpha**2)
        margin_term = torch.mean(2*alpha*(self.phat*(1-self.phat)*self.margin + self.phat*logits*neg_ind.float() - (1-self.phat)*logits*labels.float()), dim=0)
        # margin_term = 2*(1+alpha)*torch.mean((self.phat*logits*neg_ind.float() - (1-self.phat)*logits*labels.float()))
        loss = torch.sum(A1 + A2 + margin_term - cross_term)
        return loss
