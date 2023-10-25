import torch
import torch.nn as nn

class VectorQuantizer(nn.Module):

    def __init__(self, num_embeddings, embedding_dim):

        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dimensions = embedding_dim

        self.codebook = nn.Embedding(num_embeddings, embedding_dim)


    
    def forward(self, z):
        '''
        Args
            - z: latent, shape should be (B, C, H, W)
        '''

        # convert to (B, H, W, C)
        z = z.permute(0, 2, 3, 1).contiguous()
        # Flatten to (B*H*W, C)
        z_flat = z.view(-1, self.embedding_dimensions)

        distances = torch.sum(z_flat ** 2, dim=1, keepdim=True) + \
            torch.sum(self.codebook.weight**2, dim=1) - 2 * \
            torch.matmul(z_flat, self.codebook.weight.t())

        argmin_codewords = torch.argmin(distances, dim=1).unsqueeze(1)
        min_codewords = torch.zeros((argmin_codewords.shape[0], self.num_embeddings))
        min_codewords.scatter_(1, argmin_codewords, 1)

        z_quantized = torch.matmul(min_codewords, self.codebook.weight).view(z.shape).permute(0, 3, 1, 2)

        return min_codewords, z_quantized




    