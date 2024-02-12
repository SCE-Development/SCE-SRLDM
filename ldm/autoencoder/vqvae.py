# define model architecture here

from torch import Tensor
import torch.nn as nn

from encoder import Encoder
from vq import VectorQuantizer
from residual import Residual
from decoder import Decoder
from typing import Tuple


class VQVAE(nn.Module):

    def __init__(
        self,
        input_shape: Tuple[int, int],
        n_layers: int = 3,
        n_hidden: int = 128,
        n_residuals: int = 3,
        num_embeddings: int = 512,
        embedding_dim: int = 128,
        kernel_size: int = 8,
        residual_multiplier: float = 0.25,
    ):
        """
        Initializes the Vector-Quantized Variational AutoEncoder module

        Arguments:
            - input_shape: Tuple[int, int] - the H,W of the input images
            - n_layers: int - the number of encoder/decoder layers to use
            - n_hidden: int - the hidden size for encoder/decoder layers to use
            - n_residuals: int - the number of residuals to use between encoder/decoder and quantization
            - num_embeddings: int - the number of "codes" for the vector quantizer's codebook
            - embedding_dim: int - the size of each of the "codes"; should be a divisor of `n_hidden`
            - kernel_size: int - the size of the kernel to use for convolution/deconvolution
            - residual_multiplier: float - the multiplier for the residual hiddens
        """
        super(VQVAE, self).__init__()

        self.encoder = Encoder(
            input_shape, n_layers=n_layers, n_hidden=n_hidden, kernel_size=kernel_size
        )
        self.vq = VectorQuantizer(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim
        )
        self.decoder = Decoder(
            inp_shape=self.encoder.output_shape[1:],
            n_hidden=n_hidden,
            n_layers=n_layers,
            kernel_size=kernel_size,
        )

        self.residual_enc = Residual(
            n_residuals, n_hidden, kernel_size, residual_multiplier
        )
        self.residual_dec = Residual(
            n_residuals, n_hidden, kernel_size, residual_multiplier
        )

        self.to_emb = nn.Conv2d(n_hidden, embedding_dim, kernel_size, 1, "same")
        self.from_emb = nn.Conv2d(embedding_dim, n_hidden, kernel_size, 1, "same")

    def encode(self, x: Tensor) -> Tensor:
        z = self.residual_enc(self.encoder(x))

        return z

    def quantize(self, z: Tensor) -> Tensor:
        z_quantized = self.vq(z)

        return z_quantized

    def decode(self, z: Tensor) -> Tensor:
        x_hat, _ = self.decoder(self.residual_dec(z))

        return x_hat

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Call the VQVAE on the provided input.

        Arguments:
            x: Tensor - input tensor in (B,C,H,W) format

        Returns:
            Tuple[Tensor, Tensor] - a tuple containing the reconstructed image and the codebook loss
        """
        z = self.to_emb(self.residual_enc(self.encoder(x)))

        z_quantized, loss = self.vq(z)
        x_hat = self.decoder(self.residual_dec(self.from_emb(z_quantized)))

        return x_hat, loss
