import sklearn.datasets
import numpy as np
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset, DataLoader
import torch


# All toy data lies in range [-4, 4] x [-4, 4]
DATA_NAME = ['25gaussians', '8gaussians', 'swissroll', '2spirals', 'circles', '2sines', 'checkerboard', 'moon']


class Toy_Dataset(Dataset):
    def __init__(self, data_name, discrete=False):
        self.data = load_toy_data(data_name)
        self.discrete = discrete

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        single_pt = self.data[idx]
        if self.discrete:
            raise NotImplementedError
        return single_pt


def load_toy_data(data_name, data_size=100000):
    assert data_name in DATA_NAME, "Not a proper data name"
    if data_name == '25gaussians':
        centers = [-1, -.5, 0, .5, 1]
        dataset = []
        for i in range(data_size // 25):
            for x in centers:
                for y in centers:
                    point = np.random.randn(2) * 0.025
                    point += [x, y]
                    dataset.append(point)
        dataset = np.array(dataset, dtype='float32') * 2.828

    elif data_name == '8gaussians':
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1),
                   (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)),
                   (-1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (-1. / np.sqrt(2), -1. / np.sqrt(2))]
        dataset = []
        for i in range(data_size):
            point = np.random.randn(2) * 0.025
            center = random.choice(centers)
            point += center
            dataset.append(point)
        dataset = np.array(dataset, dtype='float32') * 2.828

    elif data_name == 'swissroll':
        dataset = sklearn.datasets.make_swiss_roll(n_samples=data_size, noise=0.1)[0]
        dataset = dataset.astype('float32')[:, [0, 2]] / 4.242

    elif data_name == "2spirals":
        n = np.sqrt(np.random.rand(data_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(data_size // 2, 1) * 0.1
        d1y = np.sin(n) * n + np.random.rand(data_size // 2, 1) * 0.1
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y))))
        x += np.random.randn(*x.shape) * 0.1
        dataset = x / 2.828

    elif data_name == "circles":
        data = sklearn.datasets.make_circles(n_samples=data_size, factor=.5, noise=0.025)[0]
        dataset = data.astype("float32") * 2.828

    elif data_name =="2sines":
        x = (np.random.rand(data_size) - 0.5) * 2 * np.pi
        u = (np.random.binomial(1, 0.5, data_size) - 0.5) * 2
        y = u * np.sin(x) * 2.5 + np.random.randn(*x.shape) * 0.05
        dataset =  np.stack((x, y), 1)

    elif data_name =="checkerboard":
        centers = [(-1.5, 1.5), (0.5, 1.5), (-0.5, 0.5), (1.5, 0.5), 
                    (-1.5, -0.5), (0.5, -0.5), (-0.5, -1.5), (1.5, -1.5)]
        dataset = []
        for i in range(data_size):
            point = np.random.uniform(-.5, .5, size=2)
            center = random.choice(centers)
            point += center
            dataset.append(point)
        dataset = np.array(dataset, dtype='float32') * 2

    elif data_name == "moon":
        x = np.linspace(0, np.pi, data_size // 2)
        u = np.stack([np.cos(x) + .5, -np.sin(x) + .2], axis=1)
        u += 0.025 * np.random.normal(size=u.shape)
        v = np.stack([np.cos(x) - .5, np.sin(x) - .2], axis=1)
        v += 0.025 * np.random.normal(size=v.shape)
        dataset = np.concatenate([u, v], axis=0) * 2

    return dataset

def load_noise(batch_size):
    while True:
        yield np.random.randn(batch_size, 2)


# -------- Utils for Toy plotting -------- #
def plot_toy(data, path):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    plt.clf()
    plt.xticks([])
    plt.yticks([])
    plt.scatter(data[:, 0], data[:, 1], c='cornflowerblue', marker='X')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.savefig(path)


if __name__ == '__main__':
    plot_toy(next(load_noise(2048)), 'noise.png')
    for data_name in DATA_NAME:
        dataloaer = DataLoader(Toy_Dataset(data_name, discrete=False), batch_size=2048, shuffle=True)
        plot_toy(next(iter(dataloaer)), f"{data_name}.png")