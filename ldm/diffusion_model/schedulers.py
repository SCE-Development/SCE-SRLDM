import torch
import torch.nn as nn
from abc import abstractmethod
import math


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
    def __init__(self, total_steps: int, s: float) -> None:
        super(CosineNoiseScheduler, self).__init__()
        self.total_steps = total_steps
        self.s = s

        # Register buffers for alpha_t, alpha_zero, and prev_alpha_t
        self.register_buffer('alpha_t', torch.zeros(1))
        self.register_buffer('alpha_zero', torch.zeros(1))
        self.register_buffer('prev_alpha_t', torch.zeros(1))

        # Calculate alpha_zero
        alpha_zero = math.cos((math.pi / 2) * (self.s / (1 + self.s)))
        alpha_zero = alpha_zero ** 2
        self.alpha_zero[0] = alpha_zero
        self.prev_alpha_t[0] = alpha_zero

    def calculate_alphas(self, step: int) -> None:
        """
        Calculate alpha_t, alpha_zero, and prev_alpha_t based on the squared cosine function.
        """
        assert step > 0, "step 0 is already computed"
        # Calculate alpha_t
        alpha_t = math.cos(((math.pi / 2) * step / self.total_steps + self.s) / (1 + self.s))
        alpha_t = alpha_t ** 2

        # Store the current alpha_t in prev_alpha_t
        self.prev_alpha_t[0] = self.alpha_t[0]

        # Update alpha_t
        self.alpha_t[0] = alpha_t

    def noise(self, x: torch.Tensor, step: int) -> torch.Tensor:
        """
        Custom noise scheduler based on the squared cosine function.
        """
        # Calculate alpha_t, alpha_zero, and prev_alpha_t
        self.calculate_alphas(step)

        # Calculate alpha using current and previous alpha_t
        beta = 1 - (self.alpha_t[0] / self.prev_alpha_t[0])

        return self._apply_noise(x, rate=beta)
