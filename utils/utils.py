import torch
import torch.nn as nn

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

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, group_num=16):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(group_num, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(group_num, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, factor=2,group_channel=32):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // factor, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels // factor, out_channels)
        self.norm1 = nn.GroupNorm(in_channels // factor // group_channel, in_channels // factor)
        self.norm2 = nn.GroupNorm(in_channels // factor // group_channel, in_channels // factor)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = self.norm1(x2) + self.norm2(x1)
        return self.conv(x)

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
        # self.norm = nn.GroupNorm(out_channels // group_channel, out_channels)

    def forward(self, x):
        b, c = x.shape[0], x.shape[1]
        squeeze = self.avg_pool(x).view(b, c)
        elicitation = self.linear(squeeze).view(b, c, 1, 1)
        x = x + x * elicitation.expand_as(x)
        x = self.reduce_conv(x)
        return x + self.conv(x)

class Change_Head(nn.Module):
    def __init__(self, in_channel, group_channel=32):
        super(Change_Head, self).__init__()
        self.norm_a = nn.GroupNorm(in_channel // group_channel, in_channel)
        self.norm_b = nn.GroupNorm(in_channel // group_channel, in_channel)
        self.fusion_channel_shuffle = FusionChannelShuffle(in_channel)
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channel * 2, in_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(in_channel // group_channel, in_channel),
            nn.ReLU(inplace=True)
        )
        # self.reduce = reduce_conv(in_channel*2, in_channel)

    def forward(self, align_feat):
        before, after = torch.chunk(align_feat, 2, dim=0)
        fusion = self.fusion_channel_shuffle(before, after)
        out = self.reduce(fusion)
        return out


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
