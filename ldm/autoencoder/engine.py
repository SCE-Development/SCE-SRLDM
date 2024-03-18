import torch
from torch.utils.data import DataLoader
from vqvae import VQVAE
from discriminator import Discriminator
from tqdm import tqdm
from typing import Callable, Any
from torch.utils import tensorboard
import lpips


def train_no_discriminator(
    epochs: int,
    dl: DataLoader,
    vqvae: VQVAE,
    vqvae_optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_fn: Callable[[Any], torch.Tensor] = lambda b: b,
    lper: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = lpips.LPIPS(net="vgg"),
):
    """
    Train the vqvae using purely reconstruction loss and code book losses.

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
        - lper: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] - perceptual loss to use; defaults to using LPIPS
            from https://arxiv.org/abs/1801.03924 with vgg as the net.
    """
    mse = torch.nn.MSELoss()
    lper = lper.to(device)
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
            rec_loss = (
                lper(torch.clip(reconstructed * 2, -1, 1), torch.clip(batch * 2, -1, 1))
                .mean()
                .squeeze()
            )
            full_loss = mse_loss + codebook_loss + rec_loss
            full_loss.backward()
            vqvae_optimizer.step()

            # log
            mse_loss = mse_loss.detach().cpu()
            codebook_loss = codebook_loss.detach().cpu()
            rec_loss = rec_loss.detach().cpu()
            prog.set_postfix_str(
                f"mse_loss: {mse_loss:.5f}, codebook loss: {codebook_loss:.5f}, rec_loss: {rec_loss:.5f}"
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
            torch.save(vqvae.state_dict(), "best_vqvae.pt")
            torch.save(vqvae_optimizer.state_dict(), "best_vqvae_optim.pt")
        torch.save(vqvae.state_dict(), "latest_vqvae.pt")
        torch.save(vqvae_optimizer.state_dict(), "latest_vqvae_optim.pt")


def train(
    epochs: int,
    dl: DataLoader,
    vqvae: VQVAE,
    vqvae_optimizer: torch.optim.Optimizer,
    discriminator: Discriminator,
    discriminator_optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_fn: Callable[[Any], torch.Tensor] = lambda b: b,
    lper: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = lpips.LPIPS(net="vgg"),
    gamma: float = 1e-6,
    discriminator_start_epoch: int = 80,
    disc_weight: float = 0.8,
):
    """
    Train the vqvae using gan-like perceptual training, as proposed by https://arxiv.org/pdf/2012.09841.pdf and used by Stable
    Diffusion paper.

    Arguments:
        - epochs: int - the number of epochs to train for
        - dl: DataLoader - the batched dataloader. Note: the dataloader should have the image
            tensors (in the format of C,H,W) as the first element
        - vqvae: VQVAE - the vector quantized variational autoencoder to train
        - vqvae_optimizer: torch.optim.Optimizer - the optimizer
        - discriminator: Discriminator - the patch-based discriminator that outputs logits that whose sigmoid
            is `1`s if an image is real and `0`s if it is fake
        - discriminator_optimizer: torch.optim.Optimizer - the optimizer for the discriminator
        - device: torch.device - the device to train on
        - batch_fn: Callable - a function that takes the current batch and returns the Tensor version of it.
            For most applications, an identity function would suffice. If the dataset is in HuggingFace datasets
            format, then this callback becomes useful
        - lper: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] - perceptual loss to use; defaults to using LPIPS
            from https://arxiv.org/abs/1801.03924 with vgg as the net.
        - gamma: float - small value to add to divisors to prevent division by 0
        - discriminator_start_epoch: int - when to start optimizing the vqvae using the discriminator as well as the perceptual loss
        - disc_weight: float - an extra multiplier to multiply the vqvae gan loss by (in addition to the adaptive lambda term)
    """

    def calculate_lambda(lrec_val, lgan_val):
        # lambda = ( Gradient w.r.t LL (L_rec) ) / ( Gradient w.r.t LL (L_GAN) + gamma )
        ll = vqvae.decoder.head.weight
        gwrt_lrec = torch.autograd.grad(lrec_val, ll, retain_graph=True)[0]
        gwrt_lgan = torch.autograd.grad(lgan_val, ll, retain_graph=True)[0]

        # scalar! as described in the taming-transformers codebase at
        # https://github.com/CompVis/taming-transformers/
        lambda_val = torch.norm(gwrt_lrec) / (torch.norm(gwrt_lgan) + gamma)
        lambda_val = torch.clamp(lambda_val, 0.0, 1 / gamma).detach()
        return lambda_val

    # loss functions
    bce = torch.nn.BCEWithLogitsLoss()
    lper = lper.to(device)
    l1 = torch.nn.L1Loss()

    # move models to devices
    vqvae = vqvae.train().to(device)
    discriminator = discriminator.train().to(device)

    # logging
    writer = tensorboard.SummaryWriter()
    best_rec = float("inf")

    for epoch in range(1, epochs + 1):
        # setup for logging later
        prog = tqdm(dl)
        prog.set_description_str(f"epoch {epoch}")
        epoch_disc_loss = 0
        epoch_disc_fake_loss = 0
        epoch_disc_real_loss = 0
        epoch_vqvae_loss = 0
        epoch_reconstruction_loss = 0
        epoch_gan_loss = 0
        epoch_codebook_loss = 0

        for batch in prog:
            # take only the image tensors, move them to device
            batch = batch_fn(batch)
            batch = batch.to(device)

            # generate samples
            vqvae_optimizer.zero_grad()
            reconstructed, codebook_loss = vqvae(batch)
            false_preds = discriminator(reconstructed)

            # optimize vqvae
            # lpips says to have input be scaled to [-1, 1] (so we just multiply our [-.5, .5] by 2)
            # and the taming-transformers paper adds l1 loss to the perceptual loss to create
            # their full recreation loss
            rec_loss = l1(reconstructed, batch)
            rec_loss += lper(
                torch.clip(reconstructed * 2, -1, 1), torch.clip(batch * 2, -1, 1)
            ).mean()
            rec_loss = rec_loss.flatten().squeeze()
            # adaptive lambda weight for gan loss term
            gan_loss = bce(false_preds, torch.ones_like(false_preds).to(device))
            lambda_val = calculate_lambda(rec_loss, gan_loss)
            # sum everything and backprop
            if epoch >= discriminator_start_epoch:
                vqvae_loss = (
                    rec_loss + codebook_loss + disc_weight * lambda_val * gan_loss
                )
            else:
                vqvae_loss = rec_loss + codebook_loss
            vqvae_loss.backward()
            vqvae_optimizer.step()

            # optimize discriminator
            discriminator_optimizer.zero_grad()
            true_preds = discriminator(batch)
            false_preds = discriminator(reconstructed.detach())

            # discriminator should have predicted all patches of items in batch to be 1s
            # and all patches of reconstructed items to be 0s
            discriminator_loss_real = bce(
                true_preds, torch.ones_like(true_preds).to(device)
            )
            discriminator_loss_fake = bce(
                false_preds, torch.zeros_like(false_preds).to(device)
            )
            discriminator_loss = discriminator_loss_real + discriminator_loss_fake
            discriminator_loss.backward()
            discriminator_optimizer.step()

            # log
            prog.set_postfix_str(
                f"dr: {discriminator_loss_real.detach().cpu():.5f}, df: {discriminator_loss_fake.detach().cpu():.5f}, lrec: {rec_loss.detach().cpu():.5f}, lgan: {gan_loss.detach().cpu():.5f}, lambda: {lambda_val.cpu():.5f}"
            )
            epoch_disc_loss += discriminator_loss.detach().cpu()
            epoch_disc_fake_loss += discriminator_loss_fake.detach().cpu()
            epoch_disc_real_loss += discriminator_loss_real.detach().cpu()
            epoch_vqvae_loss += vqvae_loss.detach().cpu()
            epoch_reconstruction_loss += rec_loss.detach().cpu()
            epoch_gan_loss += gan_loss.detach().cpu()
            epoch_codebook_loss += codebook_loss.detach().cpu()

        # average before logging
        epoch_disc_loss /= len(dl)
        epoch_disc_fake_loss /= len(dl)
        epoch_disc_real_loss /= len(dl)
        epoch_vqvae_loss /= len(dl)
        epoch_reconstruction_loss /= len(dl)
        epoch_gan_loss /= len(dl)
        epoch_codebook_loss /= len(dl)

        # write to tensorboard
        writer.add_scalar("discriminator/loss", epoch_disc_loss, epoch)
        writer.add_scalar("discriminator/fake_loss", epoch_disc_fake_loss, epoch)
        writer.add_scalar("discriminator/real_loss", epoch_disc_real_loss, epoch)
        writer.add_scalar(
            "discriminator/lr", discriminator_optimizer.param_groups[-1]["lr"], epoch
        )
        writer.add_scalar("vqvae/loss", epoch_vqvae_loss, epoch)
        writer.add_scalar("vqvae/reconstruction_loss", epoch_reconstruction_loss, epoch)
        writer.add_scalar("vqvae/gan_loss", epoch_gan_loss, epoch)
        writer.add_scalar("vqvae/codebook_loss", epoch_codebook_loss, epoch)
        writer.add_scalar("vqvae/lr", vqvae_optimizer.param_groups[-1]["lr"], epoch)

        # log to console as well
        print(
            f"Epoch d/loss: {epoch_disc_loss:.5f}, d/fake: {epoch_disc_fake_loss:.5f}, d/real: {epoch_disc_real_loss:.5f}, "
            f"v/loss: {epoch_vqvae_loss:.5f}, v/rec: {epoch_reconstruction_loss:.5f}, v/gan: {epoch_gan_loss:.5f}, v/code: {epoch_codebook_loss:.5f}"
        )

        # save best model as we train
        if epoch_reconstruction_loss < best_rec:
            torch.save(vqvae.state_dict(), "best_vqvae.pt")
            torch.save(vqvae_optimizer.state_dict(), "best_vqvae_optim.pt")
            torch.save(discriminator.state_dict(), "best_disc.pt")
            torch.save(discriminator_optimizer.state_dict(), "best_disc_optim.pt")

        torch.save(vqvae.state_dict(), "latest_vqvae.pt")
        torch.save(vqvae_optimizer.state_dict(), "latest_vqvae_optim.pt")
        torch.save(discriminator.state_dict(), "latest_disc.pt")
        torch.save(discriminator_optimizer.state_dict(), "latest_disc_optim.pt")
