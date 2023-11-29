import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

def show_images(images, num_images=16, size=(3, 256, 512)):
    images = images.detach().cpu().view(-1, *size)
    grid = make_grid(images[:num_images], nrow=4)
    plt.imshow(grid.permute(1, 2, 0).squeeze())
    plt.show()
