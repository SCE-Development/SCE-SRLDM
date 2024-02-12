# utililty functions for the model go here
import torch
import random
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from typing import Callable, Any


def to_image(x: torch.Tensor):
    return Image.fromarray(
        ((torch.clamp(x, -0.5, 0.5) + 0.5) * 255)
        .permute(1, 2, 0)
        .contiguous()
        .to(torch.uint8)
        .numpy()
    )


def get_comparison(
    num_comparisons: int,
    ds: Dataset,
    model: torch.nn.Module,
    batch_fn: Callable[[Any], torch.Tensor],
):
    """
    Get comparisons between real images and generated images

    Arguments:
        - num_comparisons: int - the number of comparisons to generate
        - ds: Dataset - the dataset with real images. Note: the dataset should have the image
            tensors (in the format of C,H,W) as the first element in each item
        - model: torch.nn.Module - an autoencoder
        - batch_fn:
    """
    choices = random.sample(list(range(len(ds))), num_comparisons)

    real_images = []
    for i in choices:
        real_images.append(to_image(batch_fn(ds[i])))

    reconstructed = torch.zeros((num_comparisons, *batch_fn(ds[0]).shape))
    for idx, ds_idx in enumerate(choices):
        reconstructed[idx] = batch_fn(ds[ds_idx])
    with torch.no_grad():
        reconstructed = reconstructed.to("cuda")
        reconstructed, _ = model(reconstructed)
        reconstructed = reconstructed.detach().cpu()

    reconstructed_imgs = []
    for i in range(num_comparisons):
        reconstructed_imgs.append(to_image(reconstructed[i]))

    _, axis_arr = plt.subplots(num_comparisons, 2)
    axis_arr[0, 0].set_title("Real")
    axis_arr[0, 1].set_title("Fake")
    for i in range(num_comparisons):
        axis_arr[i, 0].imshow(real_images[i])
        axis_arr[i, 1].imshow(reconstructed_imgs[i])
    plt.savefig(fname="comparisons")
