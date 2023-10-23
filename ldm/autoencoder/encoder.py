from typing import Tuple
import torch
import torch.nn as nn
import numpy as np


class EncoderUnit(nn.Module):
    """
    Encoder Unit; Halves the H,W of an input
    """

    def __init__(
        self,
        in_shape: Tuple[int, int, int],
        out_shape: Tuple[int, int, int],
        kernel_size: int,
        pool_size: int,
    ) -> None:
        super(EncoderUnit, self).__init__()

        # calculate necessary padding for the pool layer
        # H out​ =floor( (H in​ +2×padding[0]−dilation[0]×(kernel_size[0]−1)−1​)/stride[0] + 1 )
        stride = 2
        dilation = 1
        desired_height = out_shape[1]
        height_without_padding = (
            in_shape[1] + 2 * 0 - dilation * (pool_size - 1) - 1
        ) // stride + 1
        necessary_padding = (desired_height - height_without_padding) * 2 // stride

        self.unit = nn.Sequential(
            nn.Conv2d(
                in_shape[-1],
                out_shape[-1],
                kernel_size=kernel_size,
                stride=1,
                padding="same",
            ),
            nn.GELU(),
            nn.BatchNorm2d(out_shape[-1]),
            nn.Conv2d(
                out_shape[-1],
                out_shape[-1],
                kernel_size=kernel_size,
                stride=1,
                padding="same",
            ),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=pool_size, stride=2, padding=necessary_padding),
        )

    def forward(self, x: torch.Tensor):
        return self.unit(x)


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
                EncoderUnit(
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
