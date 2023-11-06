

# define model architecture here

import torch
import torch.nn as nn

from encoder import Encoder
from vq import VectorQuantizer
from decoder import Decoder


class VQVAE(nn.Module):

    def __init__(self):
        
        super(VQVAE, self).__init__()

        self.encoder = Encoder()
        self.vq = VectorQuantizer()
        self.decoder = Decoder()

        # TODO: missing a bunch of stuff here


    def encode(self, x):
        z = self.encoder(x)

        return NotImplementedError
    

    def quantize(self, z):
        z_quantized = self.vq(z)

        return NotImplementedError
    

    def decode(self, z):
        x_hat = self.decoder(z)
        
        return NotImplementedError
    
    def forward(self, x):

        z = self.encoder(x)
        z_quantized,min_code, loss = self.vq(z)
        x_hat = self.decoder(z_quantized)

        return x_hat,loss
