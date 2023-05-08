import os, sys
sys.path.append(os.getcwd()+"/Toy-Diffusion-Models")
import torch
import torch.nn as nn
from model.utils import (timestep_embedding, timesteps_to_tensor, 
                         ResBlock, Downsample, Upsample, Block)


class UNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=1, inner_channel=32, norm_groups=4,
        channel_mults=(1, 2, 2, 4), res_blocks=3, img_size=32, dropout=0):
        super().__init__()

        noise_level_channel = inner_channel
        self.time_embed_dim = inner_channel
        self.time_embed = nn.Sequential(
            nn.Linear(inner_channel, inner_channel * 4),
            nn.Mish(), 
            nn.Linear(inner_channel * 4, inner_channel)
        )

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = img_size

        # Downsampling stage of U-net
        downs = [nn.Conv2d(in_channel, inner_channel, kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResBlock(pre_channel, channel_mult, time_emb_dim=noise_level_channel, 
                                        norm_groups=norm_groups, dropout=dropout))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResBlock(pre_channel, pre_channel, time_emb_dim=noise_level_channel, 
                        norm_groups=norm_groups, dropout=dropout),
            ResBlock(pre_channel, pre_channel, time_emb_dim=noise_level_channel, 
                        norm_groups=norm_groups, dropout=dropout, att=False)
        ])

        # Upsampling stage of U-net
        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                ups.append(ResBlock(pre_channel+feat_channels.pop(), channel_mult, time_emb_dim=noise_level_channel, 
                                    norm_groups=norm_groups, dropout=dropout))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, out_channel, groups=norm_groups)

    def forward(self, x, t: int or list[int]):
        x.clamp_(-1., 1.)
        t = timesteps_to_tensor(t, batch_size=x.shape[0]).to(x.device)
        t_emb = timestep_embedding(t, self.time_embed_dim)
        t_emb = self.time_embed(t_emb)
        
        feats = []
        for layer in self.downs:
            if isinstance(layer, ResBlock):
                x = layer(x, t_emb)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            x = layer(x, t_emb)

        for layer in self.ups:
            if isinstance(layer, ResBlock):
                x = layer(torch.cat((x, feats.pop()), dim=1), t_emb)
            else:
                x = layer(x)

        return self.final_conv(x)


if __name__ == '__main__':
    model = UNet()
    x = torch.randn(2, 1, 32, 32)
    t = 1
    out = model(x, t)
    print(out.shape)