import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEFocalLoss(torch.nn.Module):
    """
    二分类的Focalloss alpha 固定
    """

    def __init__(self, gamma=1, alpha=0.75, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        alpha = self.alpha
        pt = _input.sigmoid()
        pt = torch.clamp(pt, 0.001, 0.999)
        # target = 0.9 * target + 0.1 * (1 - target)
        loss = - alpha * torch.pow(1 - pt, self.gamma) * target * torch.log(pt) - (1 - alpha) * \
               torch.pow(pt, self.gamma) * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2.0):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#
#     def forward(self, inputs, targets):
#         bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
#         pt = torch.exp(-bce_loss)
#         focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
#         return focal_loss.mean()
