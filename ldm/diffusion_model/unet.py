import torch
import torch.nn as nn
from typing import Tuple
import math

class UNet(nn.Module):
    def __init__(
        self,
        shape: int,
        latent_num_channels: int,
        channels: int,
        bottleneck_channels: int,
        kernel_size: int,
        depth: int,
        num_time_steps: int,
    ) -> None:
        """
        The UNet from stable diffusion responsible for denoising the input
        It is composed of down sampling layers with cross attention,
        a residual connection, and up sampling layers with cross attention
        """
        super(UNet, self).__init__()

        curr_channels = channels

        self.pre_conv = nn.Conv2d(
            2*latent_num_channels,
            channels,
            kernel_size=kernel_size,
            padding="same"
        )

        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        
        for _ in range(depth):
            self.down.append(
                Downsample(
                    shape,
                    curr_channels,
                    curr_channels*2,
                    kernel_size,
                    num_time_steps
                )
            )

            shape //= 2
            curr_channels *= 2

        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                curr_channels,
                bottleneck_channels,
                kernel_size=kernel_size,
                padding="same"
            ),
            nn.Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=kernel_size,
                padding="same"
            ),
            nn.Conv2d(
                bottleneck_channels,
                curr_channels,
                kernel_size=kernel_size,
                padding="same"
            )
        )

        for _ in range(depth):
            self.up.append(
                Upsample(
                    curr_channels,
                    curr_channels//2,
                    kernel_size,
                    num_time_steps,
                )
            )
            curr_channels //= 2

        self.out_conv = nn.Conv2d(
            channels + 2*latent_num_channels,
            latent_num_channels,
            kernel_size=kernel_size,
            padding="same"
        )

    def forward(self, zt: torch.Tensor, time):
        """
        Given an input zt, return zt-1
        """
        x = zt.detach().clone()
        h = []
        h.append(x)
        x = self.pre_conv(x)
        for layer in self.down:
            
            x = layer(x, time)
            h.append(x)

        x = self.bottleneck(x)

        for layer in self.up:
            x = torch.cat((x, h.pop()), dim=1)
            x = layer(x, time)

        x = torch.cat((x, h.pop()), dim=1)

        x = self.out_conv(x)
        return x
    
class Downsample(nn.Module):
    def __init__(
            self,
            shape: int,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            num_time_steps: int,
    ):
        super(Downsample, self).__init__()

        stride = 2
        out_shape = shape //2
        padding = math.ceil(((out_shape-1)*stride - shape + kernel_size)/2)

        self.num_time_steps = num_time_steps
        
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=padding
        )
        self.silu = nn.SiLU()
        self.batch_norm = nn.BatchNorm2d(
            out_channels
        )
        
        self.time_emb_layer = nn.Linear(
            1,
            out_channels
        )
        self.merged_conv = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same"
        )
        
        self.merged_batch_norm = nn.BatchNorm2d(
            out_channels
        )
            

    def forward(self, x: torch.Tensor, time: torch.Tensor):
        x = self.conv(x)
        x = self.silu(x)
        x = self.batch_norm(x)
        time_emb = self.time_emb_layer(time.unsqueeze(-1)/self.num_time_steps)
        time_emb = self.silu(time_emb)

        # convert time_emb from (B, out_channels) to (B, out_channels, 1, 1)
        x = x + time_emb.unsqueeze(-1).unsqueeze(-1)
        
        x = self.merged_conv(x)
        x = self.silu(x)
        return self.merged_batch_norm(x)
    
class Upsample(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            num_time_steps: int,
    ):
        super(Upsample, self).__init__()

        self.num_time_steps = num_time_steps

        self.conv = nn.Conv2d(
            in_channels=2*in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same"
        )
        self.silu = nn.SiLU()
        self.batch_norm = nn.BatchNorm2d(
            out_channels
        )

        self.time_emb_layer = nn.Linear(
            1,
            out_channels
        )

        self.merged_conv = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same"
        )       
        self.merged_batch_norm = nn.BatchNorm2d(
            out_channels
        )
        
    def forward(self, x: torch.Tensor, time: torch.Tensor):
        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        x = self.silu(x)
        x = self.batch_norm(x)

        time_emb = self.time_emb_layer(time.unsqueeze(-1)/self.num_time_steps)
        time_emb = self.silu(time_emb)

        # convert time_emb from (B, out_channels) to (B, out_channels, 1, 1)
        x = x + time_emb.unsqueeze(-1).unsqueeze(-1)
        x = self.merged_conv(x)
        x = self.silu(x)
        return self.merged_batch_norm(x)