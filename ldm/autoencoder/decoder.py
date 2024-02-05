from typing import Tuple
import torch
import torch.nn as nn

from units import UpUnit, ConvLayers


class Decoder(nn.Module):
    def __init__(
        self,
        inp_shape: Tuple[int, int, int],
        n_layers: int,
        n_hidden: int,
        kernel_size: int,
    ) -> None:
        """
        Initialize the decoder

        Arguments:
            - inp_shape: Tuple[int, int] - the H, W of the input compressed images
            - n_layers: int - the number of deconvolutional units to use; each unit
                doubles the H,W of the input image.
            - n_hidden: int - the hidden size to use for deconvolution
            - kernel_size: int - the size of the kernel to use for up convolution
        """
        super(Decoder, self).__init__()

        cur_shape = inp_shape
        cur_channels = n_hidden
        # add upconv layers
        for i in range(n_layers):
            next_channels = max(cur_channels // 2, 3)
            if i == n_layers - 1:
                next_channels = 3
            self.add_module(
                f"unit_{i}",
                UpUnit(cur_shape, cur_channels, next_channels, kernel_size=kernel_size),
            )
            cur_channels = next_channels
            cur_shape = (cur_shape[0] * 2, cur_shape[1] * 2)
        self.add_module(
            f"postconv",
            nn.Conv2d(3, 3, kernel_size=kernel_size, stride=1, padding="same"),
        )

    def forward(self, x: torch.Tensor):
        """
        Call the model on B,C,H,W input
        """
        for module in self.children():
            x = module(x)

        return x
