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
            self.down.append(Downsample(curr_channels, curr_channels*2))
            curr_channels *= 2

        self.bottleneck = nn.Conv2d(curr_channels, curr_channels, kernel_size=3, stride=1, padding=1)

        for i in range(depth):
            self.up.append(Upsample(curr_channels, curr_channels//2))
            curr_channels //= 2

        self.out_conv = nn.Conv2d(64, latent_num_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, zt: torch.Tensor, time):
        """
        Given an input zt, return zt-1
        """
        x = zt.detach().clone()
        h = []
        h.append(x)

        for layer in self.down:
            x = layer(x, time)
            h.append(x)

        x = self.bottleneck(x)

        for layer in self.up:
            torch.cat((x, h.pop()), dim=1)
            x = layer(x, time)

        torch.cat((x, h.pop()), dim=1)

        x = self.out_conv(x)
        return x
    
class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_shape: int = 1280):
        super(Downsample, self).__init__()

        self.silu = nn.SiLU()
        self.group_norm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        
        self.time_emb_layer = nn.Linear(time_emb_shape, out_channels)

        self.merged_group_norm = nn.GroupNorm(32, out_channels)
        self.merged_conv = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)       

    def forward(self, x: torch.Tensor, time: torch.Tensor):
        x = self.group_norm(x)
        x = self.silu(x)
        x = self.conv(x)

        time_emb = self.time_emb_layer(time)
        time_emb = self.silu(time_emb)

        # convert time_emb from (B, out_channels) to (B, out_channels, 1, 1)
        x = x + time_emb.unsqueeze(-1).unsqueeze(-1)
        x = self.merged_group_norm(x)
        x = self.silu(x)
        return self.merged_conv(x) 
class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_shape: int = 1280):
        super(Upsample, self).__init__()
        self.silu = nn.SiLU()
        self.group_norm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        
        self.time_emb_layer = nn.Linear(time_emb_shape, out_channels)

        self.merged_group_norm = nn.GroupNorm(32, out_channels)
        self.merged_conv = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)       
    
    def forward(self, x: torch.Tensor, time: torch.Tensor):
        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        x = self.group_norm(x)
        x = self.silu(x)
        x = self.conv(x)

        time_emb = self.time_emb_layer(time)
        time_emb = self.silu(time_emb)

        # convert time_emb from (B, out_channels) to (B, out_channels, 1, 1)
        x = x + time_emb.unsqueeze(-1).unsqueeze(-1)
        x = self.merged_group_norm(x)
        x = self.silu(x)
        return self.merged_conv(x)