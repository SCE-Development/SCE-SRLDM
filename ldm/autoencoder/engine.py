# code for training and inference should go here

import torch
from torch.utils.data import DataLoader
from vqvae import VQVAE
from discriminator import Discriminator
from tqdm import tqdm


def train_l2_Codebook(
    epochs: int,
    dl: DataLoader,
    vqvae: VQVAE,
    vqvae_optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    """
    Train the vqvae using reconstruction loss and code book losses.

    Arguments:
        - epochs: int - the number of epochs to train for
        - dl: DataLoader - the batched dataloader. Note: the dataloader should have the image
            tensors (in the format of C,H,W) as the first element
        - vqvae: VQVAE - the vector quantized variational autoencoder to train
        - vqvae_optimizer: torch.optim.Optimizer - the optimizer
        - device: torch.device - the device to train on
    """
    mse = torch.nn.MSELoss()
    vqvae = vqvae.train().to(device)

    for epoch in range(1, epochs + 1):
        prog = tqdm(dl)
        prog.set_description_str(f"epoch {epoch}")
        epoch_loss = 0

        for batch in prog:
            # take only the image tensors, move them to device
            batch, *_ = batch
            batch = batch.to(device)

            # generate samples
            vqvae_optimizer.zero_grad()
            reconstructed, codebook_loss = vqvae(batch)

            # optimize vqvae
            mse_loss = mse(reconstructed, batch)
            full_loss = mse_loss + codebook_loss
            full_loss.backward()
            vqvae_optimizer.step()

            # log
            mse_loss = mse_loss.detach().cpu()
            codebook_loss = codebook_loss.detach().cpu()
            prog.set_postfix_str(
                f"mse_loss: {mse_loss:.5f}, codebook loss: {codebook_loss:.5f}"
            )

            epoch_loss += full_loss.detach().cpu()

        epoch_loss /= len(dl)

        print(f"Epoch loss: {epoch_loss:.5f}")


def train(
    epochs: int,
    dl: DataLoader,
    vqvae: VQVAE,
    vqvae_optimizer: torch.optim.Optimizer,
    discriminator: Discriminator,
    discriminator_optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    """
    Train the vqvae using gan-like training.

    Arguments:
        - epochs: int - the number of epochs to train for
        - dl: DataLoader - the batched dataloader. Note: the dataloader should have the image
            tensors (in the format of C,H,W) as the first element
        - vqvae: VQVAE - the vector quantized variational autoencoder to train
        - vqvae_optimizer: torch.optim.Optimizer - the optimizer
        - discriminator: Discriminator - the discriminator that outputs `1`s if an image
            is real and `0`s if it is fake
        - discriminator_optimizer: torch.optim.Optimizer - the optimizer for the discriminator
        - device: torch.device - the device to train on
    """
    bce = torch.nn.BCELoss()

    vqvae = vqvae.train().to(device)
    discriminator = discriminator.train().to(device)

    for epoch in range(1, epochs + 1):
        prog = tqdm(dl)
        prog.set_description_str(f"epoch {epoch}")
        epoch_disc_loss = 0
        epoch_vqvae_loss = 0

        for batch in prog:
            # take only the image tensors, move them to device
            batch, *_ = batch
            batch = batch.to(device)

            # generate samples
            vqvae_optimizer.zero_grad()
            reconstructed, vqvae_loss = vqvae(batch)

            # optimize vqvae
            vqvae_loss += bce(
                discriminator(reconstructed), torch.ones((batch.shape[0], 1)).to(device)
            )
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
