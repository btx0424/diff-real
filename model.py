import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
import einops.layers.torch as ein
from diffusers import DDPMScheduler, DDIMScheduler

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
        self.conv1 = conv1d(dim=dim, stride=1)
        self.conv2 = conv1d(dim=dim, stride=1)
        self.conv_res = conv1d(dim=dim, stride=1)
        self.conv_down = conv1d(dim=dim, stride=2)

    def forward(self, x):
        res = self.conv_res(x)
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        return self.conv_down(out2 + res)

class SimpleUpBlock(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.conv1 = conv1d(dim=dim, stride=1)
        self.conv2 = conv1d(dim=dim, stride=1)
        self.up = nn.Sequential(
            nn.LazyConvTranspose1d(dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.LazyConv1d(dim * 2, kernel_size=1),
            # ein.Rearrange("b (c u) l -> b c (u l)", u=2),
            nn.GroupNorm(8, dim),
            nn.Mish(),
            nn.LazyConv1d(dim, kernel_size=3, padding=1),
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
        mean: torch.Tensor,
        std: torch.Tensor,
        time_dim: int=32,
        fourier_features: int=-1,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.time_dim = time_dim

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
        module_list.append(SimpleUpBlock(128))
        module_list.append(SimpleUpBlock(64))
        self.upsample = nn.ModuleList(module_list)

        self.initial = nn.Sequential(
            nn.LazyConv1d(64, kernel_size=3, padding=1),
        )
        self.middel = conv1d(stride=1)

        self.final = nn.Sequential(
            nn.LazyConv1d(64, kernel_size=3, padding=1),
            nn.SELU(),
            nn.LazyConv1d(output_dim, kernel_size=1),
        )
        if fourier_features > 0:
            self.register_buffer("B", 0.1 * torch.randn(fourier_features, 18))
            self.B: torch.Tensor
        
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.mean: torch.Tensor
        self.std: torch.Tensor
    
    @torch.no_grad()
    def normalize(self, x: torch.Tensor):
        return (x - self.mean.unsqueeze(-1)) / self.std.unsqueeze(-1)
    
    @torch.no_grad()
    def denormalize(self, x: torch.Tensor):
        return x * self.std.unsqueeze(-1) + self.mean.unsqueeze(-1)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Args:
            x: (B, D, T)
            t: (B,) or (B, 1)

        Returns:
            (B, output_dim, T)
        """
        t = torch.atleast_1d(t)
        if hasattr(self, "B"):
            Bx = self.B @ x
            x = torch.cat([x, torch.sin(Bx), torch.cos(Bx)], dim=1)
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

    # @torch.inference_mode()
    # def sample(self, size: int, noise_scheduler):
    #     noise_scheduler.set_timesteps(100)
        
    #     s = torch.randn(size, self.output_dim, self.time_dim, device=device)
    #     for t in noise_scheduler.timesteps:
    #         model_output = self(s, t.to(device))
    #         s = noise_scheduler.step(model_output, t, s).prev_sample
    #     return s
    
    @torch.inference_mode()
    def denoise(self, batch: torch.Tensor, noise_scheduler, steps=5):
        s = self.normalize(batch)
        noise_scheduler.set_timesteps(100)
        device = batch.device
        for t in noise_scheduler.timesteps[-steps:]:
            model_output = self(s, t.to(device).expand(batch.shape[0]))
            s = noise_scheduler.step(model_output, t, s).prev_sample
        return self.denormalize(s)
        
    @torch.inference_mode()
    def noise_denoise(self, batch: torch.Tensor, noise_scheduler, steps=5):
        s = self.normalize(batch)
        noise_scheduler.set_timesteps(100)
        device = batch.device

        noise = torch.randn_like(s)
        s = noise_scheduler.add_noise(s, noise, torch.tensor(steps))
        for t in noise_scheduler.timesteps[-steps:]:
            model_output = self(s, t.to(device).expand(batch.shape[0]))
            s = noise_scheduler.step(model_output, t, s).prev_sample
        return self.denormalize(s)


class TemporalUnetImpaint(TemporalUnet):
    
    def denoise(self, batch: torch.Tensor, cond: torch.Tensor, noise_scheduler, steps=5):
        assert batch.shape[:2] == cond.shape[:2], (batch.shape, cond.shape)
        if cond.ndim == 2:
            cond = cond.unsqueeze(-1)
        s = self.normalize(batch)
        cond = self.normalize(cond) # [N, D, T']
        
        noise_scheduler.set_timesteps(100)
        device = batch.device
        
        for t in noise_scheduler.timesteps[-steps:]:
            model_output = self(s, t.to(device).expand(batch.shape[0]))
            s = noise_scheduler.step(model_output, t, s).prev_sample
            s[:, :, :cond.shape[-1]] = cond
        return self.denormalize(s)
    
    def noise_denoise(self, batch: torch.Tensor, cond: torch.Tensor, noise_scheduler, steps=5):
        assert batch.shape[:2] == cond.shape[:2], (batch.shape, cond.shape)
        if cond.ndim == 2:
            cond = cond.unsqueeze(-1)
        s = self.normalize(batch)
        cond = self.normalize(cond) # [N, D, T']

        noise_scheduler.set_timesteps(100)
        device = batch.device

        noise = torch.randn_like(s)
        s = noise_scheduler.add_noise(s, noise, torch.tensor(steps))
        for t in noise_scheduler.timesteps[-steps:]:
            model_output = self(s, t.to(device).expand(batch.shape[0]))
            s = noise_scheduler.step(model_output, t, s).prev_sample
            s[:, :, :cond.shape[-1]] = cond
        return self.denormalize(s)

