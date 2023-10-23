from typing import Tuple
import torch
import torch.nn as nn


class ConvLayers(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        """
        Initialize the convolution layers.

        Arguments:
            - in_channels: int - the expected number of channels in the input
            - out_channels: int - the expected number of channels to output
            - kernel_size: int - the size of the kernel to use
        """
        super(ConvLayers, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding="same"),
            nn.GELU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(
                out_channels, out_channels, kernel_size, stride=1, padding="same"
            ),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor):
        """
        Call the model on B,C,H,W input
        """
        x = self.convs(x)

        return x


class DownUnit(nn.Module):
    """
    Down Unit; Halves the H,W of an input
    """

    def __init__(
        self,
        in_shape: Tuple[int, int, int],
        out_shape: Tuple[int, int, int],
        kernel_size: int,
        pool_size: int,
    ) -> None:
        """
        Initialize the DownUnit used for halving the height and width of the input

        Arguments:
            - in_shape: Tuple[int, int, int] - input shape, in H,W,C format
            - out_shape: Tuple[int, int, int] - output shape, in H,W,C format
            - kernel_size: int - the size of the kernels to use for convolution
            - pool_size: int - the size of the maxpool
        """
        super(DownUnit, self).__init__()

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
            ConvLayers(in_shape[-1], out_shape[-1], kernel_size),
            nn.MaxPool2d(kernel_size=pool_size, stride=2, padding=necessary_padding),
        )

    def forward(self, x: torch.Tensor):
        return self.unit(x)


class UpUnit(nn.Module):
    """
    Decoder Unit; Doubles the H,W of an input
    """

    def __init__(
        self,
        in_shape: Tuple[int, int, int],
        out_shape: Tuple[int, int, int],
        kernel_size: int,
    ) -> None:
        super(UpUnit, self).__init__()
        self.out_channels = out_shape[-1]

        self.head = ConvLayers(in_shape[-1], out_shape[-1], kernel_size)

        # calculate necessary padding
        # H_out=(H_in−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
        desired_h_out = in_shape[0] * 2
        h_out_without_padding = (
            (in_shape[0] - 1) * 2 - 2 * 0 + 1 * (kernel_size - 1) + 0 + 1
        )
        necessary_padding = (h_out_without_padding - desired_h_out) // 2

        self.convt = nn.ConvTranspose2d(
            out_shape[-1],
            out_shape[-1],
            kernel_size=kernel_size,
            stride=2,
            padding=necessary_padding,
        )

        self.GELU = nn.GELU()

    def forward(self, x: torch.Tensor):
        # B,C,H,W input
        # upsample by 2
        output_size = x.shape
        output_size = [
            output_size[0],
            self.out_channels,
            output_size[2] * 2,
            output_size[3] * 2,
        ]

        x = self.head(x)
        x = self.GELU(self.convt(x, output_size=output_size))
        return x
