import torch
import torch.nn.functional as F
import einops
import matplotlib.pyplot as plt
import imageio
import os
from tqdm import tqdm
from diffusers import UNet1DModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup

from dataclasses import dataclass

@dataclass
class TrainingConfig:
    image_size = 128  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 50
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

def make_dataset(path: str, seq_len = 64):
    trajs = torch.load(path)
    print(trajs)
    trajs = trajs["state_"][:, :, :3]
    
    plt.plot(trajs[0, :500, 0])
    plt.plot(trajs[0, :500, 1])
    plt.plot(trajs[0, :500, 2])
    plt.show()
    
    N, T = trajs.shape[:2]
    T = (T // seq_len) * seq_len
    trajs = trajs[:, :T].reshape(-1, seq_len, *trajs.shape[2:])
    trajs = einops.rearrange(trajs, 'b t d -> b d t')
    return trajs


def main():
    device = "cuda"

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

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

    dataset = make_dataset("/home/btx0424/lab/active-adaptation/scripts/trajs-10-24_16-27.pt")
    # dataset = x
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256)
    
    example = next(iter(dataloader))
    _, D, T = example.shape

    print(example.shape)

    from model import TemporalUnet
    model = TemporalUnet(output_dim=D).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(dataloader) * config.num_epochs),
    )

    # @torch.compile
    def train_step(batch: torch.Tensor):
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
    
    @torch.inference_mode()
    def sample(size: int):
        noise_scheduler.set_timesteps(100)
        s = torch.randn(size, D, T, device=device)
        for t in noise_scheduler.timesteps:
            model_output = model(s, t.to(device))
            s = noise_scheduler.step(model_output, t, s).prev_sample
        return s
    
    @torch.inference_mode()
    def denoise(batch: torch.Tensor):
        noise_scheduler.set_timesteps(100)
        s = batch.clone()
        for t in noise_scheduler.timesteps[-5:]:
            model_output = model(s, t.to(device).expand(batch.shape[0]))
            s = noise_scheduler.step(model_output, t, s).prev_sample
        return s
    
    os.makedirs("output", exist_ok=True)
    for epoch in range(config.num_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for i, batch in enumerate(pbar):
            loss = train_step(batch.to(device))
            lr_scheduler.step()
            
            if i % 100 == 0:
                print(loss.item())

        batch_denoised = denoise(batch.to(device)).cpu()
        ndim = 3
        fig, axes = plt.subplots(ndim, 1, figsize=(16, 6))

        for i in range(ndim):
            axes[i].plot(batch[0, i])
            axes[i].plot(batch_denoised[0, i])
        # fig, ax = plt.subplots()
        # s = sample(1).cpu()
        # ax.plot(s[0, 0])
        # ax.plot(s[0, 1])
        # ax.plot(s[0, 2])
        
        fig.savefig(f"output/{epoch}.png")
        plt.close(fig)
        

if __name__ == "__main__":
    main()