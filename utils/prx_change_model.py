import torch

from backbone.resnet import *
from utils.STSF import *
from cross_attention.x_cross import Cross_Attention


class addition_branch(nn.Module):
    def __init__(self, inchannels, reduction=4, group_channel=16, margin=0.5, factor=16, patch_size=1):
        super(addition_branch, self).__init__()
        hidden = inchannels // reduction
        self.block = nn.Sequential(
            nn.Conv2d(inchannels, hidden, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(hidden // group_channel, hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(hidden // group_channel, hidden),
            nn.ReLU(inplace=True)
        )
        self.reduce = nn.Sequential(
            nn.Conv2d(inchannels + hidden, inchannels, kernel_size=1, stride=1, bias=False),
            nn.GroupNorm(inchannels // group_channel, inchannels),
            nn.ReLU(inplace=True)
        )
        self.cont_loss = contrastive_loss(margin=margin, factor=factor)
        self.x_corss = Cross_Attention(hidden, hidden, patch_size=patch_size)

    def forward(self, input):
        branch = self.block(input)
        out = torch.cat([input, branch], dim=1)
        return self.reduce(out)


class calc_dim_loss(nn.Module):
    def __init__(self):
        super(calc_dim_loss, self).__init__()
        self.DIM = DeepInfoMaxLoss(256, 16)
        self.l0 = nn.Linear(512 * 8 * 8, 64)

    def forward(self, input):
        y = input['x4']
        B = y.shape[0] // 2
        y = self.l0(y.view(y.shape[0], -1))
        M = input['x3']
        M_prime = torch.cat((M[B + 1:], M[B].unsqueeze(0), M[1:B], M[1].unsqueeze(0)), dim=0)
        dim_loss = self.DIM(y, M, M_prime)
        return dim_loss


class Resnet_CD(nn.Module):
    def __init__(self, backbone='18', batch_stack=True):
        super(Resnet_CD, self).__init__()
        self.batch_stack = batch_stack
        if backbone == '18':
            self.backbone = resnet18()
            self.change_head = TFN(n_channels=3, n_classes=1, factor=1)
        if backbone == '34':
            self.backbone = resnet34()
            self.change_head = TFN(n_channels=3, n_classes=1, factor=1)
        if backbone == '50':
            self.backbone = resnest50(pretrained=False)
            self.change_head = TFN(n_channels=3, n_classes=1, factor=2)
        if backbone == '101':
            self.backbone = resnest101(pretrained=False)
            self.change_head = TFN(n_channels=3, n_classes=1, factor=2)

    def forward(self, before, after):
        if self.batch_stack:
            align_feat = torch.cat([before, after], dim=0)
            align_feats = self.backbone(align_feat)
        else:
            befores = self.backbone(before)
            afters = self.backbone(after)
            align_feats = []
            for i in range(4):
                align_feats.append(torch.cat([befores[i], afters[i]], dim=0))

        pred, results = self.change_head(align_feats, before)
        return pred, results
