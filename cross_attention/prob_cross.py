import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from math import sqrt
from torch.nn import LayerNorm


# from utils.masking import TriangularCausalMask, ProbMask
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.norm1 = nn.LayerNorm(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.norm2 = nn.LayerNorm(out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.drop(x)
        return x


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        # if self.mask_flag:
        #     attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
        #     scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.transpose(2, 1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn_a_b = AttentionLayer(ProbAttention(False, factor=5, attention_dropout=attn_drop, output_attention=False),
                                       d_model=dim, n_heads=num_heads, mix=False)
        self.attn_b_a = AttentionLayer(ProbAttention(False, factor=5, attention_dropout=attn_drop, output_attention=False),
                                       d_model=dim, n_heads=num_heads, mix=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, input):
        attn=None
        before_norm = self.norm1(input[0])
        after_norm = self.norm1(input[1])
        # context_b, context_a, attn = self.attn(before_norm, after_norm)
        context_b, _ = self.attn_a_b(after_norm, before_norm, before_norm, None)
        context_a, _ = self.attn_a_b(before_norm, after_norm, after_norm, None)
        before = self.norm(before_norm + self.drop_path(context_b))
        after = self.norm(after_norm + self.drop_path(context_a))
        before = before + self.drop_path(self.mlp(before))
        after = after + self.drop_path(self.mlp(after))
        return [before, after, attn]


class PatchEmbed(nn.Module):
    """ Feature to Patch Embedding
    """

    def __init__(self, inchannel, embed_dim, patch_size=16, pool=False):
        super().__init__()
        self.patch_size = patch_size
        self.pool = pool
        if pool:
            self.proj = nn.Conv2d(inchannel, embed_dim, kernel_size=1, stride=1, bias=False)
            self.pool = nn.MaxPool2d(kernel_size=patch_size, stride=patch_size)
        else:
            self.proj = nn.Conv2d(inchannel, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)

    def forward(self, x):
        x = self.proj(x)
        if self.pool:
            x = self.pool(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class PatchFeature(nn.Module):
    """ Patch Embedding to Feature
    """

    def __init__(self, inchannel, embed_dim, patch_size=16, bilinear=False):
        super().__init__()
        self.patch_size = patch_size
        self.bilinear = bilinear
        if bilinear:
            self.unsample = nn.Upsample(scale_factor=patch_size, mode='bilinear', align_corners=True)
            self.conv = nn.Conv2d(embed_dim, inchannel, kernel_size=1, stride=1, bias=True)
        else:
            if self.patch_size > 1:
                self.upsample = nn.ConvTranspose2d(embed_dim, inchannel, kernel_size=patch_size, stride=patch_size)
            else:
                self.upsample = nn.Conv2d(embed_dim, inchannel, kernel_size=1, stride=1)

    def forward(self, x):
        b, n, c = x.shape
        l = int(math.sqrt(n))
        x = x.transpose(1, 2).view(b, c, l, l).contiguous()
        if self.bilinear:
            if self.patch_size > 1:
                x = self.unsample(x)
            x = self.conv(x)
        else:
            x = self.upsample(x)
        return x


class Prob_Attention(nn.Module):
    def __init__(self, in_channel, embed_dim,
                 patch_size=16,
                 num_head=1,
                 cross_layers=1,
                 num_patches=None,
                 mlp_ratio=1.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.1,
                 attn_drop=0.1,
                 drop_path=0.1,
                 act_layer=nn.GELU,
                 norm_layer=LayerNorm,
                 temporal_embed=True):
        super(Prob_Attention, self).__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.cross_layer = cross_layers
        self.temporal_embed = temporal_embed
        self.embed = PatchEmbed(in_channel, embed_dim, patch_size, pool=False)
        self.to_feature = PatchFeature(in_channel, embed_dim, patch_size, bilinear=True)

        if self.temporal_embed:
            self.time_token_b = torch.nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
            self.time_token_a = torch.nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)

        if self.num_patches is not None:
            self.dist_token = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=True)

        blocks = []
        for _ in range(cross_layers):
            blocks.append(Block(embed_dim, num_head, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                                drop_path, act_layer, norm_layer))
        self.atten = nn.Sequential(*blocks)

        self.norm = nn.BatchNorm2d(in_channel)

    def forward(self, before, after):
        b, _, h, w = before.size()
        residual_b = before
        residual_a = after
        before = self.embed(before)
        after = self.embed(after)

        if self.temporal_embed:
            before = before + self.time_token_b.expand(b, -1, -1)
            after = after + self.time_token_a.expand(b, -1, -1)

        if self.num_patches is not None:
            before = before + self.dist_token
            after = after + self.dist_token

        if self.cross_layer > 0:
            output = self.atten([before, after, None])
            before, after, attn = output

        out_b = self.to_feature(before)
        out_a = self.to_feature(after)
        align_feat = torch.cat([out_b, out_a], dim=0)
        out_b, out_a = torch.chunk(self.norm(align_feat), 2, dim=0)
        return out_b + residual_b, out_a + residual_a, attn


if __name__ == '__main__':
    Attn = AttentionLayer(ProbAttention(False, factor=5, attention_dropout=0.0, output_attention=False),
                          d_model=512, n_heads=1, mix=False)
    q = torch.zeros([8, 256, 512])
    k = torch.zeros([8, 256, 512])
    v = torch.zeros([8, 256, 512])
    out, _ = Attn(q, k, v, None)
    print(out.size())
