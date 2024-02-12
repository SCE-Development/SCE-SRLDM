# code for training and inference should go here

import torch
from torch.utils.data import DataLoader
from vqvae import VQVAE
from discriminator import Discriminator
from tqdm import tqdm
from typing import Callable, Any
from torch.utils import tensorboard


def train_l2_Codebook(
    epochs: int,
    dl: DataLoader,
    vqvae: VQVAE,
    vqvae_optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_fn: Callable[[Any], torch.Tensor] = lambda b: b,
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
        - batch_fn: Callable - a function that takes the current batch and returns the Tensor version of it.
            For most applications, an identity function would suffice. If the dataset is in HuggingFace datasets
            format, then this callback becomes useful
    """
    mse = torch.nn.MSELoss()
    vqvae = vqvae.train().to(device)

    best = float("inf")
    writer = tensorboard.SummaryWriter()

    for epoch in range(1, epochs + 1):
        prog = tqdm(dl)
        prog.set_description_str(f"epoch {epoch}")
        epoch_loss = 0
        epoch_mse_loss = 0
        epoch_codebook_loss = 0

        for batch in prog:
            # take only the image tensors, move them to device
            batch = batch_fn(batch)
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
            epoch_mse_loss += mse_loss
            epoch_codebook_loss += codebook_loss

        # average losses
        epoch_loss /= len(dl)
        epoch_mse_loss /= len(dl)
        epoch_codebook_loss /= len(dl)

        # log scalars
        writer.add_scalar("loss/total", epoch_loss, epoch)
        writer.add_scalar("loss/mse", epoch_mse_loss, epoch)
        writer.add_scalar("loss/codebook_loss", epoch_codebook_loss, epoch)
        print(f"Epoch loss: {epoch_loss:.5f}")

        # save best
        if epoch_loss < best:
            best = epoch_loss
            torch.save(vqvae.state_dict(), "best.pt")
            torch.save(vqvae_optimizer.state_dict(), "best_optim.pt")


def train(
    epochs: int,
    dl: DataLoader,
    vqvae: VQVAE,
    vqvae_optimizer: torch.optim.Optimizer,
    discriminator: Discriminator,
    discriminator_optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_fn: Callable[[Any], torch.Tensor] = lambda b: b,
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
        - batch_fn: Callable - a function that takes the current batch and returns the Tensor version of it.
            For most applications, an identity function would suffice. If the dataset is in HuggingFace datasets
            format, then this callback becomes useful
    """
    bce = torch.nn.BCELoss()

    vqvae = vqvae.train().to(device)
    discriminator = discriminator.train().to(device)

    writer = tensorboard.SummaryWriter()

    for epoch in range(1, epochs + 1):
        prog = tqdm(dl)
        prog.set_description_str(f"epoch {epoch}")
        epoch_disc_loss = 0
        epoch_vqvae_loss = 0

        for batch in prog:
            # take only the image tensors, move them to device
            batch = batch_fn(batch)
            batch = batch.to(device)

            # generate samples
            vqvae_optimizer.zero_grad()
            reconstructed, vqvae_loss = vqvae(batch)

            # optimize discriminator
            discriminator_optimizer.zero_grad()
            true_preds = discriminator(batch)
            false_preds = discriminator(reconstructed.detach())

            # discriminator should have predicted all of items in batch to be 1s
            # and all the reconstructed items to be 0s
            discriminator_loss_real = bce(
                true_preds, torch.ones_like(true_preds).to(device)
            )
            discriminator_loss_fake = bce(
                false_preds, torch.zeros_like(false_preds).to(device)
            )
            discriminator_loss = discriminator_loss_real + discriminator_loss_fake
            discriminator_loss.backward()
            discriminator_optimizer.step()

            # optimize vqvae
            vqvae_loss += bce(
                discriminator(reconstructed), torch.ones((batch.shape[0], 1)).to(device)
            )
            vqvae_loss.backward()
            vqvae_optimizer.step()

            # log
            prog.set_postfix_str(
                f"d_loss_r: {discriminator_loss_real.detach().cpu():.5f}, d_loss_f: {discriminator_loss_fake.detach().cpu():.5f} vae_loss: {vqvae_loss.detach().cpu():.5f}"
            )

            epoch_disc_loss += discriminator_loss.detach().cpu()
            epoch_vqvae_loss += vqvae_loss.detach().cpu()

        epoch_disc_loss /= len(dl)
        epoch_vqvae_loss /= len(dl)
        writer.add_scalar("loss/disc_loss", epoch_disc_loss)
        writer.add_scalar("loss/vqvae_loss", epoch_vqvae_loss)

        print(
            f"Epoch disc loss: {epoch_disc_loss:.5f}, Epoch vqvae loss: {epoch_vqvae_loss:.5f}"
        )
