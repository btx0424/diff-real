import torch
import torch.nn.functional as F
import einops
import matplotlib.pyplot as plt
import imageio
import os
import argparse

from tqdm import tqdm
from diffusers import DDPMScheduler, DDIMScheduler, DDIMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup

from dataclasses import dataclass

@dataclass
class TrainingConfig:
    image_size = 128  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 10
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "ddpm-butterflies-128"  # the model name locally and on the HF Hub

    push_to_hub = True  # whether to upload the saved model to the HF Hub
    hub_model_id = "<your-username>/<my-awesome-model>"  # the name of the repository to create on the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

config = TrainingConfig()


class TrajDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, seq_len = 64):
        super().__init__()
        trajs = torch.load(path)
        # print(trajs)
        trajs = trajs["state_"][:, :, :3 + 12 + 12]
        # trajs[:, :, :3] /= 10
        ndim = min(trajs.shape[2], 8)
        fig, axes = plt.subplots(ndim)
        for i in range(ndim):
            axes[i].plot(trajs[0, :500, i])
        plt.show()

        N, T, D = trajs.shape

        self.seq_len = seq_len
        self.valid_starts = list(range(N * (T - self.seq_len - 1)))
        self.trajs: torch.Tensor = trajs.reshape(N * T, D)
        self.mean = self.trajs.mean(0)
        self.std = self.trajs.std(0)

    def __len__(self):
        return len(self.valid_starts)
    
    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        return self.trajs[start : start + self.seq_len].T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampler", "-s", type=str, default="ddpm")
    args = parser.parse_args()

    device = "cuda"
    
    sampler = {"ddpm": DDPMScheduler, "ddim": DDIMScheduler}[args.sampler]

    noise_scheduler: DDPMScheduler = sampler(
        num_train_timesteps=1000,
        clip_sample_range=3.0
    )

    with torch.device(device):
        N = 8192
        T = 64
        t = torch.linspace(0, torch.pi * 2, T)
        sin_t = torch.sin(t).expand(N, T)
        cos_t = torch.cos(t).expand(N, T)
        sin2p1 = torch.square(sin_t) + 1
        c = torch.rand(8192, 1)
        x = torch.stack([cos_t, sin_t * cos_t, c * sin_t], dim=-1)
        x = x / sin2p1.unsqueeze(-1)

        x = einops.rearrange(x, 'b t d -> b d t')
        print(x.shape)

    dataset = TrajDataset("/home/btx0424/lab/active-adaptation/scripts/trajs-11-11_20-48.pt")
    # dataset = x
    dataloader_train = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)
    
    example = next(iter(dataloader_train))
    _, D, T = example.shape

    print(example.shape)

    from model import TemporalUnet, TemporalUnetImpaint
    model = TemporalUnetImpaint(D, dataset.mean, dataset.std).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(dataloader_train) * config.num_epochs),
    )

    # @torch.compile
    def train_step(batch: torch.Tensor):
        batch = model.normalize(batch)
        bs = batch.shape[0]
        noise = torch.randn_like(batch)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=batch.device)
        noisy_images = noise_scheduler.add_noise(batch, noise, timesteps)

        noise_pred = model(noisy_images, timesteps)
        loss = F.mse_loss(noise_pred, noise)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss
    
    os.makedirs("output", exist_ok=True)
    for epoch in range(config.num_epochs):
        pbar = tqdm(dataloader_train, desc=f"Epoch {epoch}")
        for i, batch in enumerate(pbar):
            loss = train_step(batch.to(device))
            lr_scheduler.step()
            if i % 100 == 0:
                pbar.set_postfix({"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]})

        if epoch % 1 == 0:
            # error = 0
            # for i, batch in enumerate(dataloader_eval):
            #     batch = batch.to(device)
            #     batch_denoised = denoise(batch)
            #     error += einops.reduce((batch - batch_denoised).square(), "b d t -> d", "mean")
            # error = error / len(dataloader_eval)
            # print(error.tolist())
            batch = batch.to(device)
            cond = batch[:, :, 0].unsqueeze(-1)
            batch_denoised = model.denoise(batch, cond, noise_scheduler)
            batch_noised_denoised = model.noise_denoise(batch, cond, noise_scheduler)
            ndim = min(batch.shape[1], 18)
            fig, axes = plt.subplots(ndim, 1, figsize=(16, 15))

            for i in range(ndim):
                axes[i].plot(batch[0, i].cpu(), label="original")
                axes[i].plot(batch_denoised[0, i].cpu(), label="denoised")
                axes[i].plot(batch_noised_denoised[0, i].cpu(), label="noised denoised")
                axes[i].legend()

            fig.suptitle(f"Epoch {epoch}")
            fig.savefig(f"output/{epoch}.png")
            plt.close(fig)
    
        torch.save(model, f"ckpt-{epoch}.pt")

if __name__ == "__main__":
    main()