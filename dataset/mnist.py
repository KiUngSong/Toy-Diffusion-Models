import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms


class MNIST_Dataset(Dataset):
    def __init__(self, data, discrete=False):
        self.data = data
        self.discrete = discrete

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, _ = self.data[idx]
        if self.discrete:
            img = torch.round(img)
        return img

def load_mnist_data(discrete=False):
    transform = [transforms.Resize(32), transforms.ToTensor()]
    if not discrete:
        transform.append(transforms.Normalize((0.5,), (0.5,)))

    data = torchvision.datasets.MNIST('./data', transform=transforms.Compose(transform), train=True, download=True)
    return MNIST_Dataset(data, discrete=discrete)

if __name__ == "__main__":
    dataloader = DataLoader(load_mnist_data(), batch_size=100, shuffle=False)
    imgs = next(iter(dataloader))
    torchvision.utils.save_image(imgs.detach().cpu(), f'mnist.jpg', nrow=10, value_range=(-1,1))

    dataloader = DataLoader(load_mnist_data(discrete=True), batch_size=100, shuffle=False)
    imgs = next(iter(dataloader))
    torchvision.utils.save_image(imgs.detach().cpu(), f'mnist_discrete.jpg', nrow=10, value_range=(0,1))