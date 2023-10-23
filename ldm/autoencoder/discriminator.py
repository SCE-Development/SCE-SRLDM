

import torch
import torch.nn as nn

class Discriminator(nn.Module):

    def __init__(self):

        super(Discriminator, self).__init__()

        self.conv_layers = nn.Sequential(
            # TODO: some number of nn.Conv2d's, BatchNorms, and LeakyRELU's
        )

    def forward(self, x):
        pred = self.conv_layers(x)
        return pred
    