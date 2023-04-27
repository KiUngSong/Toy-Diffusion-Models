from .mnist import load_mnist_data
from .toy_2d import Toy_Dataset, DATA_NAME, plot_toy

from torch.utils.data import DataLoader
import torchvision.utils as vutils
from PIL import Image
import numpy as np
import os


def get_dataloader(data_name: str, discrete: bool, batch_size: int, num_workers: int):
    if data_name in DATA_NAME:
        dataset = Toy_Dataset(data_name, discrete)
    elif data_name == "mnist":
        dataset = load_mnist_data(discrete)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


#--------------------  Utils for Visualization --------------------#

def save2img(img, path, data_type="img", discrete=False):
    if data_type == 'toy' and not discrete:
        plot_toy(img, path)
    elif discrete:
        vutils.save_image(img.float(), path, normalize=True, value_range=(0,1), nrow=int(np.sqrt(img.size(0))))
    else:
        vutils.save_image(img, path, normalize=True, value_range=(-1,1), nrow=int(np.sqrt(img.size(0))))

def make_gif(plot_paths, git_name):
        frames = [Image.open(fn) for fn in plot_paths]
        frames[0].save(os.path.join(f"{git_name}.gif"), format='GIF', append_images=frames[1:], 
                       save_all=True, duration=75, loop=0)