import torch
import torch.nn as nn

class VectorQuantizer(nn.Module):

    def __init__(self, num_embeddings, embedding_dim):

        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dimensions = embedding_dim

        self.codebook = nn.Embedding(num_embeddings, embedding_dim)


    '''
    foward pass

    INPUTS:
    z - latent, shape should be (B, C, H, W)
    '''
    def forward(self, z):
        

        return NotImplementedError




    