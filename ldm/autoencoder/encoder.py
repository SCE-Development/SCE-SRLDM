from typing import Tuple
import torch
import torch.nn as nn
import numpy as np

from units import DownUnit


class Encoder(nn.Module):
    def __init__(
        self,
        inp_shape: Tuple[int, int, int],
        out_shape: Tuple[int, int, int],
        kernel_size: int,
        pool_size: int,
    ) -> None:
        """
        Initialize the encoder

        Arguments:
            - inp_shape: Tuple[int, int] - the H, W, C of the input images
            - out_shape: Tuple[int, int] - the H, W, C of the output
            - kernel_size: int - the size of the convolution kernels
            - pool_size: int - the size of the MaxPool kernels
        """
        super(Encoder, self).__init__()

        cur_shape = np.array(inp_shape)
        out_shape = np.array(out_shape)
        i = 0

        # add downconv layers
        while (cur_shape != out_shape).all():
            desired_shape = np.array(
                [cur_shape[0] // 2, cur_shape[1] // 2, out_shape[-1]]
            )
            if cur_shape[-1] * 2 <= desired_shape[-1]:
                desired_shape[-1] = cur_shape[-1] * 2
            self.add_module(
                f"unit_{i}",
                DownUnit(
                    cur_shape,
                    desired_shape,
                    kernel_size=kernel_size,
                    pool_size=pool_size,
                ),
            )

            cur_shape = desired_shape
            i += 1

    def forward(self, x: torch.Tensor):
        """
        Call the model on B,C,H,W input
        """
        for module in self.children():
            x = module(x)

        return x
