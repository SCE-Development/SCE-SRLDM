import torch
import torch.nn as nn
from abc import abstractmethod


class NoiseScheduler(nn.Module):
    def __init__(self, t: type):
        super(t, self).__init__()

    @abstractmethod
    def noise(self, x: torch.Tensor, step: int):
        """
        Returns the noised version of x according to the
        step `step`

        Arguments:
            - x: torch.Tensor - the tensor to run the forward
                diffusion noise process on
            - step: int - the current step that the forward
                diffusion noise process is on
        """

    def _apply_noise(self, x: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
        """
        q(xt|xt-1) := N (xt; sqrt(1 - βt)xt-1, βtI)

        Returns q(xt|xt-1)

        Arguments:
            - x: torch.Tensor - the tensor to apply the noise to
            - rate: torch.Tensor - the beta to use
        """
        # technically it should be elementwise, but we can basically
        # do it with U(sqrt(1-Bt)xt-1) + N(0, BtI)
        xt = torch.sqrt(1 - rate) * x + torch.normal(0, rate, x.shape)
        return xt


class ConstantNoiseScheduler(NoiseScheduler):
    def __init__(self, beta: float) -> None:
        super().__init__(ConstantNoiseScheduler)
        self.register_buffer("beta", torch.tensor(beta))

    def noise(self, x: torch.Tensor, step: int) -> torch.Tensor:
        """
        q(xt|xt-1) := N (xt; sqrt(1 - βt)xt-1, βtI)
        """
        return self._apply_noise(x, rate=self.beta)


class CosineNoiseScheduler(NoiseScheduler):
    def __init__(self, t: type):
        super().__init__(CosineNoiseScheduler)

        # TODO - get Vishwesh's cosine noise scheduler here
        pass

    def noise(self, x: torch.Tensor, step: int) -> torch.Tensor:
        pass
