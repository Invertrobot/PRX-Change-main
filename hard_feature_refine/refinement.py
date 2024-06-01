import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.resnet import *
from utils.utils import *
from config import opt
from hard_feature_refine.sampling_points import sampling_points, point_sample


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.3):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv1d(in_features, hidden_features, 1)
        self.norm = nn.LayerNorm(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv1d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class refine_head(nn.Module):
    def __init__(self, in_c=65, k=int(opt.k), beta=float(opt.beta), stride=None):
        super().__init__()
        self.mlp = Mlp(in_c, in_c, 1)
        self.k = k
        self.beta = beta
        self.stride = stride

    def forward(self, x, feat, proj):
        num_point = x.shape[-1] // self.stride
        hard_points = sampling_points(proj, num_point, self.k, self.beta)

        features = point_sample(feat, hard_points, align_corners=False)
        points = point_sample(proj, hard_points, align_corners=False)

        hard_features = torch.cat([features, points], dim=1)

        RP = self.mlp(hard_features)

        return {"RP": RP, "hard_points": hard_points}
