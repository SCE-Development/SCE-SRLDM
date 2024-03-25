from typing import Tuple, Union, List
import torch
import torch.nn as nn

from units import DownUnit


class Encoder(nn.Module):
    def __init__(
        self,
        inp_shape: Tuple[int, int],
        n_layers: int,
        n_hidden: int,
        kernel_size: int,
        downsample_factor: Union[int, List[int]] = 2,
    ) -> None:
        """
        Initialize the encoder

        Arguments:
            - inp_shape: Tuple[int, int] - the H, W of the input images
            - n_layers: int - the number of convolution units to use; each convolution unit halves
                the H,W of the input.
            - n_hidden: int - the hidden size to use for convolution
            - kernel_size: int - the size of the convolution kernels
            - downsample_factor: int | List[int] - the downsample factor to use for each of the layers. If an integer is provided,
            then that downsample factor is used for all layers. Defaults to 2
        """
        super(Encoder, self).__init__()

        if type(downsample_factor) == int:
            downsample_factor = [downsample_factor] * n_layers

        cur_channels = 3
        cur_shape = inp_shape
        self.units = nn.ModuleList()

        for i in range(n_layers):
            next_channels = max(cur_channels * 2, n_hidden // (2 ** (n_layers - i - 1)))
            self.units.add_module(
                f"{i}",
                DownUnit(
                    cur_shape,
                    cur_channels,
                    next_channels,
                    kernel_size=kernel_size,
                    downsample_factor=downsample_factor[i],
                ),
            )
            cur_channels = next_channels
            cur_shape = (
                cur_shape[0] // downsample_factor[i],
                cur_shape[1] // downsample_factor[i],
            )

        self.output_shape = (n_hidden, *cur_shape)  # C,H,W

    def forward(self, x: torch.Tensor):
        """
        Call the model on B,C,H,W input
        """
        for unit in self.units:
            x = unit(x)
        return x
