from typing import Tuple
import torch
import torch.nn as nn
import numpy as np


class DecoderHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        """
        Decoder Unit Head; runs convolution
        """
        super(DecoderHead, self).__init__()

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


class DecoderUnit(nn.Module):
    """
    Decoder Unit; Doubles the H,W of an input
    """

    def __init__(
        self,
        in_shape: Tuple[int, int, int],
        out_shape: Tuple[int, int, int],
        kernel_size: int,
    ) -> None:
        super(DecoderUnit, self).__init__()
        self.out_channels = out_shape[-1]

        self.decoder_head = DecoderHead(in_shape[-1], out_shape[-1], kernel_size)

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

        x = self.decoder_head(x)
        x = self.GELU(self.convt(x, output_size=output_size))
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        inp_shape: Tuple[int, int, int],
        out_shape: Tuple[int, int, int],
        kernel_size: int,
        head_size: int,
    ) -> None:
        """
        Initialize the decoder

        Arguments:
            - inp_shape: Tuple[int, int] - the H, W, C of the input images
            - out_shape: Tuple[int, int] - the H, W, C of the output
        """
        super(Decoder, self).__init__()

        cur_shape = np.array(inp_shape)
        out_shape = np.array(out_shape)
        i = 0

        # add upconv layers
        while (cur_shape != out_shape).all():
            desired_shape = np.array(
                [cur_shape[0] * 2, cur_shape[1] * 2, out_shape[-1]]
            )
            if cur_shape[-1] // 2 >= out_shape[-1]:
                desired_shape[-1] = cur_shape[-1] // 2
            self.add_module(
                f"unit_{i}",
                DecoderUnit(cur_shape, desired_shape, kernel_size=kernel_size),
            )

            cur_shape = desired_shape
            i += 1

        # add output head
        self.add_module(f"unit_{i}", DecoderHead(3, 3, head_size))

    def forward(self, x: torch.Tensor):
        """
        Call the model on B,C,H,W input
        """
        for module in self.children():
            x = module(x)

        return x
