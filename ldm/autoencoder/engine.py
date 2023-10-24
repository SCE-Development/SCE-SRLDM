# code for training and inference should go here

import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from vqvae import VQVAE
from discriminator import Discriminator
from tqdm import tqdm


def train(
    epochs: int,
    dl: DataLoader,
    vqvae_: VQVAE,
    vqvae_optimizer: torch.optim.Optimizer,
    discriminator: Discriminator,
    discriminator_optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    bce = torch.nn.BCELoss()
    mse = torch.nn.MSELoss()  # optional mse to use as well when training

    vqvae_ = vqvae_.train().to(device)
    discriminator = discriminator.train().to(device)

    for epoch in range(1, epochs + 1):
        prog = tqdm(dl)
        prog.set_description_str(f"epoch {epoch}")
        epoch_disc_loss = 0
        epoch_vqvae_loss = 0

        for batch in prog:
            batch, _ = batch
            batch = batch.to(device)

            # generate samples
            vqvae_optimizer.zero_grad()
            reconstructed = vqvae_(batch)

            # optimize vqvae
            vqvae_loss = bce(
                discriminator(reconstructed), torch.ones((batch.shape[0], 1)).to(device)
            )
            vqvae_loss += mse(reconstructed, batch)
            vqvae_loss.backward()
            vqvae_optimizer.step()

            # optimize discriminator
            discriminator_optimizer.zero_grad()
            true_preds = discriminator(batch)
            false_preds_d = discriminator(reconstructed.detach())

            # discriminator should have predicted all of items in batch to be 1s
            # and all the reconstructed items to be 0s
            discriminator_loss_real = bce(
                true_preds, torch.ones_like(true_preds).to(device)
            )
            discriminator_loss_fake = bce(
                false_preds_d, torch.zeros_like(false_preds_d).to(device)
            )
            discriminator_loss = (discriminator_loss_real + discriminator_loss_fake) / 2
            discriminator_loss.backward()
            discriminator_optimizer.step()

            # optimize the vqvae

            # log
            prog.set_postfix_str(
                f"d_loss_r: {discriminator_loss_real.detach().cpu():.5f}, d_loss_f: {discriminator_loss_fake.detach().cpu():.5f} vae_loss: {vqvae_loss.detach().cpu():.5f}"
            )

            epoch_disc_loss += discriminator_loss.detach().cpu()
            epoch_vqvae_loss += vqvae_loss.detach().cpu()

        epoch_disc_loss /= len(dl)
        epoch_vqvae_loss /= len(dl)

        print(
            f"Epoch disc loss: {epoch_disc_loss:.5f}, Epoch vqvae loss: {epoch_vqvae_loss:.5f}"
        )


if __name__ == "__main__":
    pass
