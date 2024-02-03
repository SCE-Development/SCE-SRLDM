import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(
        self, latent_num_channels, depth: int
    ) -> None:
        """
        The UNet from stable diffusion responsible for denoising the input
        It is composed of down sampling layers with cross attention,
        a residual connection, and up sampling layers with cross attention
        """
        super(UNet, self).__init__()

        self.pre_conv = nn.Conv2d(latent_num_channels, 64, kernel_size=3, stride=1, padding=1)
        curr_channels = 64

        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        
        for i in range(depth):
            self.down.insert(Downsample(curr_channels))
            # attention / residual
            curr_channels *= 2

        self.bottleneck = nn.Linear() # TODO change to something more appropriate

        for i in range(depth):
            self.up.insert(Upsample(curr_channels))
            # attention /residual
            curr_channels /= 2

        self.out_conv = nn.Conv2d(64, latent_num_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, zt: torch.Tensor):
        """
        Given an input zt, return zt-1
        """
        # TODO - actually run zt through model
        x = zt.detach().clone()
        h = []
        h.append(x)

        for i, layer in enumerate(self.down):
            x = layer(x)
            h.append(x)

        x = self.bottleneck(x)

        for i, layer in enumerate(self.up):
            x = layer(x)
            torch.cat((x, h.pop()), dim=1)

        x = self.out_conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, outchannels):
        super(Downsample, self).__init__()
    

    def foward(self, x):
        return "encoded x"

class Upsample(nn.Module):
     def __init__(self, in_channels, outchannels):
        super(Upsample, self).__init__()

    def forward(self, x, residual):
    
        return "decoded x"
