import torch
import torch.nn as nn
from typing import Tuple, Union
from schedulers import NoiseScheduler
from unet import UNet


class DiffusionModel(nn.Module):
    def __init__(
        self,
        down_layer: type,
        up_layer: type,
        depth: int,
        steps: int,
        scheduler: NoiseScheduler,
    ) -> None:
        """
        The full diffusion model
        """
        super(DiffusionModel, self).__init__()
        self.scheduler = scheduler
        self.steps = steps

        # create U-net
        self.u_net = UNet(down_layer_t=down_layer, up_layer_t=up_layer, depth=depth)

    def forward_diffusion(
        self, x: torch.Tensor, return_steps: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Runs the forward diffusion process on the provided input, returning the
        noised input. If `return_steps`, then also return the input at each step

        Arguments:
            - x - torch.Tensor - the input to run the forward diffusion process on
            - return_steps: bool - whether or not to also return the input x at
                each step t = [0, T] in the diffusion process
        """
        if return_steps:
            steps = [x]

        # add noise
        for step in range(steps):
            x = self.scheduler.noise(x, step)
            if return_steps:
                steps.append(x)

        # return result
        if return_steps:
            return x, steps
        return x

    def backwards_diffusion(
        self, x: torch.Tensor, return_steps: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Runs the backwards diffusion process on the provided input, returning the denoised input.
        If `return_steps`, then also return the input at each step

        Arguments:
            - x - torch.Tensor - the input to run the backwards diffusion process on
            - return_steps: bool - whether or not to also return the input x at
                each step t = [T, 0] in the diffusion process
        """
        if return_steps:
            steps = [x]

        # denoise
        for _ in range(steps):
            x = self.u_net(x)
            if return_steps:
                steps.append(x)

        # return result
        if return_steps:
            return x, steps
        return x

    def forward(
        self, x: torch.Tensor, return_loss: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Runs the full diffusion process on the provided input, returning the reconstructed
        input. If `return_loss` is true, then also return the loss for the diffusion model
        """

        if return_loss:
            zt, forward_steps = self.forward_diffusion(x)
            z0, backward_steps = self.backwards_diffusion(zt)
            loss = torch.nn.L1Loss()
        else:
            z0 = self.backwards_diffusion(self.forward_diffusion(x, False), False)

        if return_loss:
            l = 0
            forward_steps = forward_steps[:-1]
            backward_steps = backward_steps[:-1]

            assert forward_steps.shape == backward_steps.shape

            for t in range(forward_steps.shape[0]):
                l += loss(forward_steps[t], backward_steps[t])

            return z0, loss
        return z0
