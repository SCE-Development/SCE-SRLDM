from typing import Tuple
import torch
import torch.nn as nn
import numpy as np

from units import DownUnit


class Encoder(nn.Module):
    def __init__(
        self,
        inp_shape: Tuple[int, int],
        n_layers: int,
        n_hidden: int,
        kernel_size: int,
    ) -> None:
        """
        Initialize the encoder

        Arguments:
            - inp_shape: Tuple[int, int] - the H, W of the input images
            - n_layers: int - the number of convolution units to use; each convolution unit halves
                the H,W of the input.
            - n_hidden: int - the hidden size to use for convolution
            - kernel_size: int - the size of the convolution kernels
        """
        super(Encoder, self).__init__()

        cur_channels = 3
        cur_shape = inp_shape
        for i in range(n_layers):
            next_channels = max(cur_channels * 2, n_hidden // (2 ** (n_layers - i - 1)))
            self.add_module(
                f"unit_{i}",
                DownUnit(
                    cur_shape,
                    cur_channels,
                    next_channels,
                    kernel_size=kernel_size,
                ),
            )
            cur_channels = next_channels
            cur_shape = (cur_shape[0] // 2, cur_shape[1] // 2)

        self.output_shape = (n_hidden, *cur_shape)  # C,H,W

    def forward(self, x: torch.Tensor):
        """
        Call the model on B,C,H,W input
        """
        for module in self.children():
            x = module(x)

        return x
