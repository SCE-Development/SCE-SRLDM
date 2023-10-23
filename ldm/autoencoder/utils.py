# utililty functions for the model go here
import torch
import random
import matplotlib.pyplot as plt
from PIL import Image


def to_image(x: torch.Tensor):
    return Image.fromarray(
        (torch.clamp(x, 0, 1) * 255)
        .permute(1, 2, 0)
        .contiguous()
        .to(torch.uint8)
        .numpy()
    )


def get_comparison(num_comparisons: int, images: torch.Tensor, model: torch.nn.Module):
    choices = []
    while len(choices) != num_comparisons:
        i = random.randint(0, len(images) - 1)
        if i not in choices:
            choices.append(i)

    real_images = []
    for i in choices:
        real_images.append(to_image(images[i]))

    reconstructed = torch.zeros((num_comparisons, 3, 32, 32))
    for idx, ds_idx in enumerate(choices):
        reconstructed[idx] = images[ds_idx]
    reconstructed = reconstructed.to("cuda")
    reconstructed = model(reconstructed).detach().cpu()

    reconstructed_imgs = []
    for i in range(num_comparisons):
        reconstructed_imgs.append(to_image(reconstructed[i]))

    _, axis_arr = plt.subplots(num_comparisons, 2)
    for i in range(num_comparisons):
        axis_arr[i, 0].imshow(real_images[i])
        axis_arr[i, 1].imshow(reconstructed_imgs[i])
    plt.savefig(fname="comparisons")
