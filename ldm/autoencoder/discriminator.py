import torch
import torch.nn as nn
from typing import Tuple
import numpy as np

from units import DownUnit


class Discriminator(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        n_conv_layers: int,
        n_ff_layers: int,
        kernel_size: int,
        pool_size: int,
    ):
        """
        Initialize the discriminator. The discriminator will output
        0 if it is fake and 1 if it is real.

        Arguments:
            - input_shape: Tuple[int, int, int] - The input shape, in H,W,C
            - n_conv_layers: int - the number of convolutional layers to use
            - n_ff_layers: int - the number of feedforward layers to use after flattening
            - kernel_size: int - the size of the convolution kernels
            - pool_size: int - the size of the maxpool kernels
        """
        super(Discriminator, self).__init__()

        # add convolutional layers
        self.conv_layers = nn.Sequential()
        cur_shape = np.array(input_shape)
        for i in range(n_conv_layers):
            desired_shape = cur_shape.copy()
            desired_shape[0:2] //= 2
            desired_shape[2] *= 2
            self.conv_layers.add_module(
                str(i + 1), DownUnit(cur_shape, desired_shape, kernel_size, pool_size)
            )
            cur_shape = desired_shape

        # add linear layers
        linear_inp_shape = np.prod(cur_shape)
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
        x = self.conv_layers(x)
        x = x.flatten(start_dim=1)
        return self.head(x)
