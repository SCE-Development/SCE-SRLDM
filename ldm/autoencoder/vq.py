import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as func


class VectorQuantizer(nn.Module):

    def __init__(self, num_embeddings, embedding_dim):

        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dimensions = embedding_dim
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z: Tensor):
        """
        Args
            - z: latent, shape should be (B, C, H, W)
        """

        # store the original inputs
        original_z = z

        # convert to (B, H, W, C)
        z = z.permute(0, 2, 3, 1).contiguous()
        # Flatten to (B*H*W, C)
        z_flat = z.view(-1, self.embedding_dimensions)

        # Calculate distances between z and codebook
        distances = torch.sum((z_flat.unsqueeze(1) - self.codebook.weight) ** 2, dim=2)

        argmin_codewords = torch.argmin(distances, dim=1).unsqueeze(1)
        min_codewords = torch.zeros(
            (argmin_codewords.shape[0], self.num_embeddings), device=z.device
        )
        min_codewords.scatter_(1, argmin_codewords, 1)

        z_quantized = (
            torch.matmul(min_codewords, self.codebook.weight)
            .view(z.shape)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        # calculating codebook losses
        # move enc towards vq and move vq towards enc
        enc2quantized = func.mse_loss(z_quantized.detach(), original_z)
        quantized2enc = 0.25 * func.mse_loss(z_quantized, original_z.detach())
        loss = enc2quantized + quantized2enc

        # make it as if the quantizer "didn't exist" by removing the quantizer gradients
        # thereby allowing encoder to also receive gradients from down the line
        z_quantized = original_z + (z_quantized - original_z).detach()
        return z_quantized, loss
