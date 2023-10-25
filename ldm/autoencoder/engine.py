# code for training and inference should go here

import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from vqvae import VQVAE
from discriminator import Discriminator
from tqdm import tqdm
from torchvision.datasets import CIFAR10
import torchvision.transforms as t
from utils import get_comparison


def train_l2(
    epochs: int,
    dl: DataLoader,
    vqvae: VQVAE,
    vqvae_optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    """
    Train the vqvae using only l2 loss

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
        epoch_mse_loss = 0

        for batch in prog:
            # take only the image tensors, move them to device
            batch, *_ = batch
            batch = batch.to(device)

            # generate samples
            vqvae_optimizer.zero_grad()
            reconstructed = vqvae(batch)

            # optimize vqvae
            mse_loss = mse(reconstructed, batch)
            mse_loss.backward()
            vqvae_optimizer.step()

            # log
            mse_loss = mse_loss.detach().cpu()
            prog.set_postfix_str(f"mse_loss: {mse_loss:.5f}")

            epoch_mse_loss += mse_loss

        epoch_mse_loss /= len(dl)

        print(f"Epoch mse loss: {epoch_mse_loss:.5f}")


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
            reconstructed = vqvae(batch)

            # optimize vqvae
            vqvae_loss = bce(
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


parser = ArgumentParser()
parser.add_argument("-epochs", type=int, default=1)
parser.add_argument("-method", type=str, default="gan-like", choices=["gan-like", "l2"])
parser.add_argument("-vqvae_lr", type=float, default=1e-3)
parser.add_argument("-discriminator_lr", type=float, default=3e-4)
parser.add_argument("-batch_size", type=int, default=128)
args = parser.parse_args()


if __name__ == "__main__":
    # instantiate models
    vqvae = VQVAE()
    # arbitrary discriminator parameters that can be changed
    discriminator = Discriminator((32, 32, 3), 2, 4, 4, 2)

    # instantiate optimizers
    vqvae_optim = torch.optim.AdamW(vqvae.parameters(), args.vqvae_lr)
    discriminator_optim = torch.optim.AdamW(
        discriminator.parameters(), args.discriminator_lr
    )

    # get dataset
    ds = CIFAR10(
        root=".",
        download=True,
        transform=t.Compose(
            [
                t.PILToTensor(),
                t.Lambda(lambda x: x.to(torch.float32) / 255),
            ]
        ),
    )
    dl = DataLoader(ds, batch_size=args.batch_size)

    # use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train
    if args.method == "gan-like":
        train(
            args.epochs,
            dl,
            vqvae,
            vqvae_optim,
            discriminator,
            discriminator_optim,
            device,
        )
    elif args.method == "l2":
        train_l2(args.epochs, dl, vqvae, vqvae_optim, device)
    else:
        print("ERROR: unknown train method")

    # save comparison
    get_comparison(4, ds, vqvae)
