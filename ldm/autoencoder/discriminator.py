import torch
import torch.nn as nn
from typing import Tuple
import numpy as np

from encoder import Encoder
from residual import Residual


class Discriminator(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int],
        n_conv_layers: int,
        n_res_layers: int,
        n_hidden: int,
        kernel_size: int,
        res_multiplier: float = 0.5,
    ):
        """
        Initialize the patch-based discriminator. For every patch in the input
        image, the discriminator will output in the continuous range `[0, 1]`
        where numbers closer to `0` mean fake and numbers closer to `1` mean real.

        Arguments:
            - input_shape: Tuple[int, int] - The input shape, in H,W
            - n_conv_layers: int - the number of convolutional layers to use
            - n_res_layers: int - the number of residual layers to use after encoding
            - n_hidden: int - the size of the hidden dimension for the convolutions
            - kernel_size: int - the size of the convolution kernels
            - res_multiplier: float - the multiplier to use for each of the residual layers
        """
        super(Discriminator, self).__init__()

        self.enc = Encoder(input_shape, n_conv_layers, n_hidden, kernel_size)

        # add residual blocks
        self.residual = Residual(n_res_layers, n_hidden, kernel_size, res_multiplier)

        # head
        self.head = nn.Sequential(
            nn.Conv2d(n_hidden, 1, kernel_size, 1, padding="same"), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        """
        Call model on B,C,H,W input
        """
        x = self.residual(self.enc(x))
        x = self.head(x).squeeze()

        return x
