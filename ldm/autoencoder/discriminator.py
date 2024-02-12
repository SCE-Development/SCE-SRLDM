import torch
import torch.nn as nn
from typing import Tuple
import numpy as np

from encoder import Encoder


class Discriminator(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int],
        n_conv_layers: int,
        n_ff_layers: int,
        n_hidden: int,
        kernel_size: int,
    ):
        """
        Initialize the discriminator. The discriminator will output
        0 if it is fake and 1 if it is real.

        Arguments:
            - input_shape: Tuple[int, int, int] - The input shape, in H,W,C
            - n_conv_layers: int - the number of convolutional layers to use
            - n_ff_layers: int - the number of feedforward layers to use after flattening
            - n_hidden: int - the size of the hidden dimension for the convolutions
            - kernel_size: int - the size of the convolution kernels
        """
        super(Discriminator, self).__init__()

        self.enc = Encoder(input_shape, n_conv_layers, n_hidden, kernel_size)

        # add linear layers
        linear_inp_shape = np.prod(self.enc.output_shape)
        self.head = nn.Sequential()
        for i in range(n_ff_layers):
            self.head.add_module(
                str(i + 1), torch.nn.Linear(linear_inp_shape, linear_inp_shape)
            )
            self.head.add_module(str(i + 1) + "_gelu", torch.nn.GELU())
        # add classification head
        self.head.add_module(str(i + 1), torch.nn.Linear(linear_inp_shape, 1))
        self.head.add_module("output_func", torch.nn.Sigmoid())

    def forward(self, x: torch.Tensor):
        """
        Call model on B,C,H,W input
        """
        x = self.enc(x)
        x = x.flatten(start_dim=1)
        return self.head(x)
