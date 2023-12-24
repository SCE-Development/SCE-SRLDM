import torch
import torch.nn as nn
from abc import abstractmethod


class NoiseScheduler(nn.Module):
    @abstractmethod
    def noise(self, x: torch.Tensor, step: int):
        """
        Returns the noised version of x according to the
        step `step`

        Arguments:
            - x: torch.Tensor - the tensor to run the forward
                diffusion noise process on
            - step: int - the zero-indexed current step that the forward
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
        super(ConstantNoiseScheduler, self).__init__()
        self.register_buffer("beta", torch.tensor(beta))

    def noise(self, x: torch.Tensor, step: int) -> torch.Tensor:
        """
        q(xt|xt-1) := N (xt; sqrt(1 - βt)xt-1, βtI)
        """
        return self._apply_noise(x, rate=self.beta)


class LinearNoiseScheduler(NoiseScheduler):
    def __init__(self, b1: float, b2: float, total_steps: int) -> None:
        super(LinearNoiseScheduler, self).__init__()
        assert total_steps > 1, "There must be more than 1 step"
        self.register_buffer("increment", torch.tensor((b2 - b1) / (total_steps - 1)))
        self.register_buffer("b1", torch.tensor(b1))

    def noise(self, x: torch.Tensor, step: int):
        return self._apply_noise(x, step * self.increment + self.b1)


class CosineNoiseScheduler(NoiseScheduler):
    def __init__(self):
        super(CosineNoiseScheduler, self).__init__()

        # TODO - get Vishwesh's cosine noise scheduler here
        pass

    def noise(self, x: torch.Tensor, step: int) -> torch.Tensor:
        pass
