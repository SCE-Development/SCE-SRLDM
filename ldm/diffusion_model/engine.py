import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as t
from tqdm import tqdm
from .unet import UNet
from .schedulers import NoiseScheduler
from typing import Callable, Any

def train_unet(
        img_shape: int,
        epochs: int,
        num_timesteps: int,
        dl: DataLoader,
        unet: UNet,
        noise_scheduler: NoiseScheduler,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        batch_fn: Callable[[Any], torch.Tensor] = lambda b: b,
    ):

    unet.train()
    unet.to(device=device)

    mse = nn.MSELoss()
    
    total_loss = 0

    downscale_factor = 4
    low_res_size = img_shape//downscale_factor
    low_res_scaler = t.Resize(size=low_res_size)
    nearest_upscaler = t.Resize(img_shape, interpolation=t.InterpolationMode.NEAREST_EXACT)

    for epoch in range(1, epochs + 1):
        prog = tqdm(dl)
        prog.set_description_str(f"epoch {epoch}")
        
        epoch_loss = 0

        for batch in prog:

            batch = batch_fn(batch)

            batch = batch.to(device)

            # downscale truth
            # resize down_scaled to be same size as truth           
            # same size as truth but with lower detail
            low_res = nearest_upscaler(low_res_scaler(batch)).to(device)

            # generate random time steps for each image
            time_steps = torch.randint(low=0, high=num_timesteps, size=(batch.shape[0],), device=device)

            # generate noise from time step
            noise = torch.randn(batch.shape, device=device).to(device)

            # add noise to low_res
            noisy_images = noise_scheduler.noise(batch, noise, time_steps)

            # condition inputs with low_res by concating on channels
            noisy_images = torch.cat([noisy_images, low_res], dim=1)

            # run low res through model
            predicted_noise = unet(noisy_images, time_steps)
            
            # calculate loss through mse of predicted noise and actual noise
            loss = mse(predicted_noise, noise)

            # update gradients from loss
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().cpu()
            epoch_loss += loss

            prog.set_postfix_str(
                f"mse_loss: {loss:.5f}"
            )
        epoch_loss /= len(dl)
        print(f"Epoch loss: {epoch_loss:.7f}")

        # save best
        if epoch_loss < best:
            best = epoch_loss
            torch.save(unet.state_dict(), "./models/best.pt")
            torch.save(optimizer.state_dict(), "./models/best_optim.pt")
        
