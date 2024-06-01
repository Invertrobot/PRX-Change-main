""" Full assembly of the parts to form the complete network """
import nntplib
import random

import torch

import torch.nn.functional as F
from utils.utils import *
from config import opt
from cross_attention.x_cross import Cross_Attention
from hard_feature_refine.refinement import refine_head
import numpy as np


class FusionChannelShuffle(nn.Module):
    def __init__(self, num_groups):
        super(FusionChannelShuffle, self).__init__()
        self.num_groups = num_groups

    def forward(self, feat1, feat2, differ=None):
        batch_size, chs, h, w = feat1.shape
        chs_per_group = chs // self.num_groups
        feat1 = torch.reshape(feat1, (batch_size, self.num_groups, chs_per_group, h, w))
        feat2 = torch.reshape(feat2, (batch_size, self.num_groups, chs_per_group, h, w))
        if differ is not None:
            differ = torch.reshape(differ, (batch_size, self.num_groups, chs_per_group, h, w))
            fusion = torch.cat([feat1, feat2, differ], dim=2)
        else:
            fusion = torch.cat([feat1, feat2], dim=2)
        fusion = fusion.transpose(1, 2)
        out = torch.reshape(fusion, (batch_size, -1, h, w))
        return out


class reduce_conv(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=32, group_channel=32):
        super().__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(out_channels // group_channel, out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(out_channels // group_channel, out_channels),
            nn.ReLU(inplace=True)
        )
        self.linear = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        b, c = x.shape[0], x.shape[1]
        squeeze = self.avg_pool(x).view(b, c)
        elicitation = self.linear(squeeze).view(b, c, 1, 1)
        x = x + x * elicitation.expand_as(x)
        x = self.reduce_conv(x)
        return x + self.conv(x)


class tri_fusion(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size=2, group_channel=32):
        super().__init__()
        self.first = (in_channels == out_channels)
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        if not self.first:
            self.conv = reduce_conv(out_channels * 3, out_channels)
        else:
            self.conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, stride=1)
        self.fusion_channel_shuffle = FusionChannelShuffle(out_channels)
        self.norm = nn.GroupNorm(out_channels // group_channel, out_channels)

        if opt.cross:
            self.x_cross = Cross_Attention(out_channels, out_channels, patch_size=patch_size)

    def forward(self, align_feat, differ=None):
        attn = None
        before, after = torch.chunk(align_feat, 2, dim=0)
        if opt.cross:
            before, after = self.x_cross(before, after)
        before = self.norm(before)
        after = self.norm(after)

        if not self.first:
            differ = self.upsample(differ)
            differ = self.norm(differ)
            # out = torch.cat([before, after, differ], dim=1)
            out = self.fusion_channel_shuffle(before, after, differ)
            out = self.conv(out)
        else:
            # out = torch.cat([before, after], dim=1)
            out = self.fusion_channel_shuffle(before, after)
            out = self.conv(out)

        return out


class PFRM(nn.Module):
    def __init__(self, factor, n_classes):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Conv2d(512 * factor, n_classes, 1),
            nn.Conv2d(256 * factor, n_classes, 1),
            nn.Conv2d(128 * factor, n_classes, 1),
            nn.Conv2d(64 * factor, n_classes, 1)
        ])
        self.refine_head = nn.ModuleList([
            refine_head(512 * factor + 1, stride=32),
            refine_head(256 * factor + 1, stride=16),
            refine_head(128 * factor + 1, stride=8),
            refine_head(64 * factor + 1, stride=4),
        ])

    def forward(self, x, fusion):
        refines = []
        proj_outs = []
        for i in range(0, len(fusion)):
            proj = self.proj[i](fusion[i])
            refines.append(self.refine_head[i](x, fusion[i], proj))
            proj_outs.append(proj)
        return refines, proj_outs


class TFN(nn.Module):
    def __init__(self, n_channels, n_classes, factor=1):
        super(TFN, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.tf0 = tri_fusion(512 * factor, 512 * factor, patch_size=1)  # 16*16
        self.tf1 = tri_fusion(512 * factor, 256 * factor, patch_size=1)  # 32*32
        self.tf2 = tri_fusion(256 * factor, 128 * factor, patch_size=2)  # 64*64
        self.tf3 = tri_fusion(128 * factor, 64 * factor, patch_size=4)  # 128*128

        self.PFRM = PFRM(factor, n_classes)

        self.upsample1 = nn.ConvTranspose2d(64 * factor, 32 * factor, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(32 * factor, 32 * factor, kernel_size=2, stride=2)
        self.outc = OutConv(32 * factor, n_classes)

    def forward(self, align_feats, x=None):
        results = {'refines': []}

        fusion0 = self.tf0(align_feats[3], None)
        fusion1 = self.tf1(align_feats[2], fusion0)
        fusion2 = self.tf2(align_feats[1], fusion1)
        fusion3 = self.tf3(align_feats[0], fusion2)

        if self.training and opt.refine:
            refines, proj_outs = self.PFRM(x, [fusion0, fusion1, fusion2, fusion3])
            results.update({'refines': refines})
            results.update({'proj_outs': proj_outs})

        up = self.upsample1(fusion3)
        c = self.upsample2(up)
        pred = self.outc(c)

        return pred, results
