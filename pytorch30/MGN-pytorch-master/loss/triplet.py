#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F
from loss.lossywj import hard_example_mining, euclidean_dist
from torch.autograd import Variable

class MSMLLoss(nn.Module):
    """
    Shape:
        - Input: :math:`(N, C)` where `C = number of channels`
        - Target: :math:`(N)`
        - Output: scalar.
    """

    def __init__(self, margin=0.3, size_average=True):
        super(MSMLLoss, self).__init__()
        self.margin = margin
        self.size_average = size_average
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        #inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        print(dist_ap.size())
        print(dist_an.size())

        print(dist_ap.size())
        print(dist_an.size())
        y = Variable(dist_an.data.new().resize_as_(dist_an.data).fill_(1))
        print(y)
        loss2 = self.ranking_loss(dist_an, dist_ap, y)
        print(loss2)
        return loss2

class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3, mutual_flag = False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        #inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        if self.mutual:
            return loss, dist
        return loss

 
 
class CenterLoss(nn.Module):
    """Center loss.
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
 
    def __init__(self,use_gpu=True):
        super(CenterLoss, self).__init__()  
        self.use_gpu = use_gpu

 
    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        
        num_classes = labels.size(0)
        feat_dim = x.size(1)
        batch_size = x.size(0)
 
        if self.use_gpu:
            centers = nn.Parameter(torch.randn(num_classes, feat_dim).cuda())
        else:
            centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, num_classes) + \
                  torch.pow(centers, 2).sum(dim=1, keepdim=True).expand(num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, centers.t())
 
        classes = torch.arange(num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, num_classes)
        mask = labels.eq(classes.expand(batch_size, num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss
