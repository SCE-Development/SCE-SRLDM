import torch.nn as nn
from torch import Tensor


class ResidualUnit(nn.Module):
    def __init__(self, n_channels: int, kernel_size: int, multiplier: float):
        """
        Initialize the Residual Unit, which is just several consecutive convolutions
        summed together

        Arguments:
            - n_channels: int - the number of channels initially
            - kernel_size: int - the size of the kernel to use
            - multiplier: float - the size of the hidden layer, with respect to `n_channels`
        """
        super(ResidualUnit, self).__init__()

        self.unit = nn.Sequential(
            nn.Conv2d(
                n_channels,
                int(n_channels * multiplier),
                kernel_size,
                padding="same",
                stride=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                int(n_channels * multiplier),
                n_channels,
                kernel_size,
                padding="same",
                stride=1,
            ),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.unit(x)


class Residual(nn.Module):
    def __init__(
        self, n_layers: int, n_channels: int, kernel_size: int, multiplier: float = 0.25
    ):
        """
        Initialize a Residual

        Arguments:
            - n_layers: int - the number of residual connections to use
            - n_channels: int - the number of channels in the input
            - kernel_size: int - the size of the kernel to use
            - multiplier: float - the multiplier to use for each residual layer; defaults to `2`
        """
        super(Residual, self).__init__()

        self.sequential = nn.Sequential(
            *[
                ResidualUnit(n_channels, kernel_size, multiplier)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.sequential(x)
