import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, drop_rate=0.0):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop_rate)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):

    def __init__(self, dim, heads=8, dim_head=64, drop_rate=0.0):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = (1 / heads) ** 0.5
        self.to_qkv = nn.Linear(dim, 3 * inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim), nn.Dropout(drop_rate)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv
        )
        attn = F.softmax(
            torch.matmul(q, k.transpose(-1, -2)) * self.scale, dim=-1
        )
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, drop_rate=0.0):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(dim), Attention(dim, heads=heads, dim_head=dim_head, drop_rate=drop_rate)
                ),
                nn.Sequential(
                    nn.LayerNorm(dim), FeedForward(dim, mlp_dim, drop_rate=drop_rate)
                )
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):

    def __init__(self, size, dim, depth, heads, mlp_dim,
                 channels=3, dim_head=8, drop_rate=0.0, emb_drop_rate=0.0):
        super(ViT, self).__init__()
        num_patches = size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
            nn.Linear(channels, dim)
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.val_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_drop_rate)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, drop_rate)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, 1)
        )
        self.val_head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, 1, bias=True)
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)
        val_tokens = repeat(self.val_token, '() n d -> b n d', b=x.size(0))
        x = torch.cat([val_tokens, x], dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x)
        return self.mlp_head(x[:, 1:]).squeeze(-1), self.val_head(x[:, 0]).squeeze(-1)


def ViT4_16_8(size=15, drop_rate=0.0):
    depth = 4
    dim = 16
    heads = 8
    return ViT(size=size, dim=dim, depth=depth, heads=heads,
               mlp_dim=2 * dim, dim_head=dim // heads, drop_rate=drop_rate)


def ViT4_32_8(size=15, drop_rate=0.0):
    depth = 4
    dim = 32
    heads = 8
    return ViT(size=size, dim=dim, depth=depth, heads=heads,
               mlp_dim=2 * dim, dim_head=dim // heads, drop_rate=drop_rate)


def ViT4_64_8(size=15, drop_rate=0.0):
    depth = 4
    dim = 64
    heads = 8
    return ViT(size=size, dim=dim, depth=depth, heads=heads,
               mlp_dim=2 * dim, dim_head=dim // heads, drop_rate=drop_rate)


def ViT6_16_4(size=15, drop_rate=0.0):
    depth = 6
    dim = 16
    heads = 4
    return ViT(size=size, dim=dim, depth=depth, heads=heads,
               mlp_dim=2 * dim, dim_head=dim // heads, drop_rate=drop_rate)


def ViT6_16_8(size=15, drop_rate=0.0):
    depth = 6
    dim = 16
    heads = 8
    return ViT(size=size, dim=dim, depth=depth, heads=heads,
               mlp_dim=2 * dim, dim_head=dim // heads, drop_rate=drop_rate)


def ViT6_32_4(size=15, drop_rate=0.0):
    depth = 6
    dim = 32
    heads = 4
    return ViT(size=size, dim=dim, depth=depth, heads=heads,
               mlp_dim=2 * dim, dim_head=dim // heads, drop_rate=drop_rate)


def ViT6_32_8(size=15, drop_rate=0.0):
    depth = 6
    dim = 32
    heads = 8
    return ViT(size=size, dim=dim, depth=depth, heads=heads,
               mlp_dim=2 * dim, dim_head=dim // heads, drop_rate=drop_rate)


def ViT6_64_4(size=15, drop_rate=0.0):
    depth = 6
    dim = 64
    heads = 4
    return ViT(size=size, dim=dim, depth=depth, heads=heads,
               mlp_dim=2 * dim, dim_head=dim // heads, drop_rate=drop_rate)


def ViT6_64_8(size=15, drop_rate=0.0):
    depth = 6
    dim = 64
    heads = 8
    return ViT(size=size, dim=dim, depth=depth, heads=heads,
               mlp_dim=2 * dim, dim_head=dim // heads, drop_rate=drop_rate)


def ViT8_64_8(size=15, drop_rate=0.0):
    depth = 8
    dim = 64
    heads = 8
    return ViT(size=size, dim=dim, depth=depth, heads=heads,
               mlp_dim=2 * dim, dim_head=dim // heads, drop_rate=drop_rate)