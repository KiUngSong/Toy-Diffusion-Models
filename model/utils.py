import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Timestep Embedding
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
        timesteps: [N] dimensional tensor of int.
        dim: the dimension of the output.
        max_period: controls the minimum frequency of the embeddings.
        return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def timesteps_to_tensor(ts: int or list[int], batch_size):
    if isinstance(ts, list):
        assert batch_size % len(ts) == 0, "batch_size must be divisible by length of timesteps list"
    
    if isinstance(ts, int):
        return ts * torch.ones(batch_size)
    else:
        mini_batch_size = batch_size // len(ts)
        return torch.cat([ts[i] * torch.ones(mini_batch_size) for i in range(len(ts))])
    

# Utils for MNIST model
class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))

class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            nn.Mish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class SelfAtt(nn.Module):
    def __init__(self, channel_dim, num_heads, norm_groups=4):
        super(SelfAtt,self).__init__()        
        self.groupnorm = nn.GroupNorm(norm_groups, channel_dim)
        self.channel_dim = channel_dim
        self.num_heads = num_heads
        self.qkv = nn.Conv2d(channel_dim, channel_dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(channel_dim, channel_dim, 1)

    def forward(self,x):
        b, c, h, w = x.size()
        x = self.groupnorm(x)
        qkv = self.qkv(x).view(b, 3, self.num_heads, c // self.num_heads, h, w).permute(1, 0, 2, 3, 4, 5).contiguous()
        qkv = qkv.view(3, b,  self.num_heads, c // self.num_heads, h * w)
        queries, keys, values = qkv[0], qkv[1], qkv[2]

        keys = F.softmax(keys, dim=-1)
        att = torch.einsum('bhdn,bhen->bhde', keys, values)
        out = torch.einsum('bhde,bhdn->bhen', att, queries)
        out = out.view(b, self.num_heads, c // self.num_heads, h, w).contiguous()
        out = out.view(b, c, h, w)

        return self.proj(out)


class ResBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, norm_groups=32, num_heads=8, dropout=0, att=True):
        super().__init__()
        self.mlp = nn.Sequential(nn.Mish(), nn.Linear(time_emb_dim, dim_out))
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.att = att
        self.attn = SelfAtt(dim_out, num_heads=num_heads, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        y = self.block1(x)
        y += self.mlp(time_emb).view(x.shape[0], -1, 1, 1)
        y = self.block2(y)
        x = y + self.res_conv(x)
        if self.att:
            x = self.attn(x)
        return x


if __name__ == '__main__':
    # Check dimension of timestep embedding if get confused
    t = 1
    t_emb = timestep_embedding(t * torch.ones(32), 128)
    print(t_emb.shape)
    
    ts = [1, 2, 3, 4]
    ts = timesteps_to_tensor(ts, 16)
    t_emb = timestep_embedding(ts, 128)
    print(t_emb.shape)

    # Check dimension of self-attention if get confused
    attn = SelfAtt(128, 8)
    x = torch.randn(4, 128, 32, 32)
    y = attn(x)
    print(y.shape)
