from typing import Tuple
import torch
import torch.nn as nn


class DownUnit(nn.Module):
    """
    Down Unit; Decreases the H,W of an input by a factor of downsample_factor
    """

    def __init__(
        self,
        in_shape: Tuple[int, int],
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        downsample_factor: int = 2,
    ) -> None:
        """
        Initialize the DownUnit used for halving the height and width of the input

        Arguments:
            - in_shape: Tuple[int, int] - input shape, in H,W format
            - in_channels: int - the number of channels in the input
            - out_channels: int - the number of channels in the output
            - kernel_size: int - the size of the kernels to use for convolution
            - downsample_factor: int - how much to downsample by, defaults to 2
        """
        super(DownUnit, self).__init__()

        # calculate necessary padding for the pool layer
        # H out​ =floor( (H in​ +2×padding[0]−dilation[0]×(kernel_size[0]−1)−1​)/stride[0] + 1 )
        stride = downsample_factor
        dilation = 1
        desired_height = in_shape[0] // stride
        height_without_padding = (
            in_shape[0] + 2 * 0 - dilation * (kernel_size - 1) - 1
        ) // stride + 1
        necessary_padding = (
            (desired_height - height_without_padding) * stride + 1
        ) // 2

        self.unit = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=downsample_factor,
                padding=necessary_padding,
            ),
            nn.GELU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor):
        """
        Call the model on B,C,H,W input
        """
        return self.unit(x)


class UpUnit(nn.Module):
    """
    UpUnit; Doubles the H,W of an input
    """

    def __init__(
        self,
        in_shape: Tuple[int, int],
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        upscale_factor: int = 2,
    ) -> None:
        """
        Initialize the UpUnit

        Arguments:
            - in_shape: Tuple[int, int] - the expected input shape, in H,W format
            - in channels:int
            - out_channels:int
            - kernel_size: int - the size of the kernel to use for the convolutional transpose
        """
        super(UpUnit, self).__init__()
        self.out_channels = out_channels

        # calculate necessary padding
        # H_out=(H_in−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
        self.upscale_factor = upscale_factor
        stride = upscale_factor
        desired_h_out = in_shape[0] * stride
        h_out_without_padding = (
            (in_shape[0] - 1) * stride - 2 * 0 + 1 * (kernel_size - 1) + 0 + 1
        )
        necessary_padding = ((h_out_without_padding - desired_h_out) + 1) // 2

        self.convt = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=necessary_padding,
        )

        self.GELU = nn.GELU()

    def forward(self, x: torch.Tensor):
        """
        Call the model on B,C,H,W input
        """
        # upsample by 2
        output_size = x.shape
        output_size = [
            output_size[0],  # B
            self.out_channels,  # C
            output_size[2] * self.upscale_factor,  # H
            output_size[3] * self.upscale_factor,  # W
        ]

        x = self.GELU(self.convt(x, output_size=output_size))
        return x
