import torch
import torch.nn as nn
from typing import Tuple, Union, List
from schedulers import NoiseScheduler, LinearNoiseScheduler
from unet import UNet


class DiffusionModel(nn.Module):
    def __init__(
        self,
        u_net: UNet,
        T: int = 1000,
        scheduler: NoiseScheduler = LinearNoiseScheduler(1e-4, 2e-2, 1000),
    ) -> None:
        """
        The full diffusion model

        Arguments:
            - u_net: UNet - the UNet to use for denoising
            - T: int - the number of diffusion steps to use
            - scheduler: NoiseScheduler - the NoiseScheduler to use for the forward
                diffusion process
        """
        super(DiffusionModel, self).__init__()
        self.scheduler = scheduler
        self.T = T
        self.u_net = u_net

    def forward_diffusion(
        self, x: torch.Tensor, return_steps: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Runs the forward diffusion process on the provided input, returning the
        noised input. If `return_steps`, then also return the input at each step

        Arguments:
            - x - torch.Tensor - the input to run the forward diffusion process on
            - return_steps: bool - whether or not to also return the input x at
                each step t = [0, T] in the diffusion process

        Returns:
            - Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]] - one of the following:
                - torch.Tensor - just the noised input
                - Tuple[torch.Tensor, List[torch.Tensor]] - a tuple containing the noised
                    input and a list of the noised input at stages t = [0, T]
        """
        if return_steps:
            steps = [x]

        # add noise
        for step in range(self.T):
            x = self.scheduler.noise(x, step)
            if return_steps:
                steps.append(x)

        # return result
        if return_steps:
            return x, steps
        return x

    def backwards_diffusion(
        self, x: torch.Tensor, return_steps: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Runs the backwards diffusion process on the provided input, returning the denoised input.
        If `return_steps`, then also return the input at each step

        Arguments:
            - x - torch.Tensor - the input to run the backwards diffusion process on
            - return_steps: bool - whether or not to also return the input x at
                each step t = [T, 0] in the diffusion process

        Returns:
            - Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]] - one of the following:
                - torch.Tensor - just the reconstructed input
                - Tuple[torch.Tensor, List[torch.Tensor]] - a tuple containing the reconstructed
                    input and a list of the reconstructed input at stages t = [T, 0]
        """
        if return_steps:
            steps = [x]

        # denoise
        for _ in range(self.T):
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

        Arguments:
            - x: torch.Tensor - the input to run the diffusion process on
            - return_loss: bool - whether or not to return the diffusion model's
                loss along with the reconstructed input

        Returns:
            - Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] - one of the following:
                - torch.Tensor - just the reconstructed input
                - Tuple[torch.Tensor, torch.Tensor] - a tuple containing the reconstructed
                    input and the diffusion model's loss
        """

        if return_loss:
            zt, forward_steps = self.forward_diffusion(x)
            z0, backward_steps = self.backwards_diffusion(zt)
            loss = torch.nn.MSELoss()
        else:
            z0 = self.backwards_diffusion(self.forward_diffusion(x, False), False)

        if return_loss:
            l = 0
            forward_steps = forward_steps[:-1]
            backward_steps = backward_steps[:-1]

            for t in range(len(forward_steps)):
                l += loss(forward_steps[t], backward_steps[t])

            return z0, l
        return z0
