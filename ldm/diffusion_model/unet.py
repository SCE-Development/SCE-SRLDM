import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(
        self, up_layer_t: nn.Module, down_layer_t: nn.Module, depth: int
    ) -> None:
        """
        The UNet from stable diffusion responsible for denoising the input
        It is composed of down sampling layers with cross attention,
        a residual connection, and up sampling layers with cross attention
        """
        super(UNet, self).__init__()
        pass

    def forward(self, zt: torch.Tensor):
        """
        Given an input zt, return zt-1
        """
        pass
