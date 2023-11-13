import torch


class LinearNoiseScheduler:
    def __init__(self, start_linear, end_linear, max_epochs, noise_factor):
        self.start_linear = start_linear
        self.end_linear = end_linear
        self.max_epochs = max_epochs
        self.noise_factor = noise_factor
        self.epoch = 0

    def get_linearNoise(self):
        # Calculate the current learning rate based on linear schedule
        lr = self.initial_lr + (self.end_lr - self.initial_lr) * (self.epoch / self.max_epochs)

        # Introduce noise to the learning rate
        lr += torch.randn(1).item() * self.noise_factor

        return lr

    def step(self):
        # Increment the epoch counter
        self.epoch += 1


# Example usage:
start_linear = 0.1
end_linear = 0.01
max_epochs = 50
noise_factor = 0.01
