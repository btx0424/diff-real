import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


def conv1d(dim: int = 64, stride: int = 1):
    return nn.Sequential(
        nn.LazyConv1d(dim, kernel_size=3, stride=stride, padding=1),
        nn.GroupNorm(8, dim),
        nn.Mish(),
    )


class SimpleDownBlock(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.conv1 = conv1d(dim=dim, stride=2)
        self.conv2 = conv1d(dim=dim, stride=1)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        return out1 + out2

class SimpleUpBlock(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.conv1 = conv1d(dim=dim, stride=1)
        self.conv2 = conv1d(dim=dim, stride=1)
        self.up = nn.Sequential(
            nn.LazyConvTranspose1d(64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, 64),
            nn.Mish(),
            nn.LazyConv1d(64, kernel_size=3, padding=1),
            nn.Mish(),
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = out1 + out2
        out4 = self.up(out3)
        return out4


class TemporalUnet(nn.Module):
    def __init__(
        self, 
        output_dim: int,
        time_dim: int=32,
    ) -> None:
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )

        module_list = []
        
        module_list.append(SimpleDownBlock(64))
        module_list.append(SimpleDownBlock(128))
        module_list.append(SimpleDownBlock(128))
        self.downsample = nn.ModuleList(module_list)

        module_list = []
        module_list.append(SimpleUpBlock(128))
        module_list.append(SimpleUpBlock(64))
        module_list.append(SimpleUpBlock(64))
        self.upsample = nn.ModuleList(module_list)

        self.initial = nn.Sequential(
            nn.LazyConv1d(64, kernel_size=3, padding=1),
        )
        self.middel = conv1d(stride=1)

        self.final = nn.Sequential(
            nn.LazyConv1d(64, kernel_size=3, padding=1),
            nn.SELU(),
            nn.LazyConv1d(output_dim, kernel_size=3, padding=1),
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Args:
            x: (B, D, T)
            t: (B,) or (B, 1)

        Returns:
            (B, output_dim, T)
        """
        t = torch.atleast_1d(t)
        x = self.initial(x)
        time_emb = einops.repeat(self.time_mlp(t), "b d -> b d t", t=x.shape[-1])
        x = torch.cat([x, time_emb], dim=1)

        downsampled = []
        for downsample in self.downsample:
            downsampled.append(x)
            x = downsample(x)

        x = self.middel(x)

        for upsample, downsample in zip(self.upsample, reversed(downsampled)):
            # x = upsample(x) + downsample
            x = torch.cat([upsample(x), downsample], dim=1)
        
        x = self.final(x)
        return x
        
