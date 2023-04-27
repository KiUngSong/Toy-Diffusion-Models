import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from model import ToyNet, UNet
from dataset import get_dataloader, DATA_NAME
from diffusion.ddpm import DDPMProcess
from diffusion.score_sde import SDEProcess
from diffusion.discrete import DiscreteProcess


class DiffusionModule(pl.LightningModule):
    def __init__(self,
                 data_name: str,
                 diffusion_type: str,
                 discrete: bool,
                 total_steps: int,
                 lr: float,
                 num_discrete: int = 2,
                 sde_type: str = "VP",
                 ):
        super(DiffusionModule, self).__init__()
        self.data_name = data_name
        self.diffusion_type = diffusion_type
        self.total_steps = total_steps
        self.discrete = discrete
        self.num_discrete = num_discrete
        self.lr = lr

        # Diffusion process
        if self.diffusion_type == "ddpm":
            self.diffusion = DDPMProcess(discrete=discrete, total_steps=total_steps)
        elif self.diffusion_type == "sde":
            self.diffusion = SDEProcess(discrete=discrete, total_steps=total_steps, sde_type=sde_type)
        elif self.diffusion_type == "discrete":
            self.diffusion = DiscreteProcess(discrete=discrete, total_steps=total_steps, num_discrete=num_discrete)
        else:
            raise NotImplementedError

        if self.data_name in DATA_NAME and not self.discrete:
            self.net = ToyNet()
        elif self.data_name == "mnist" and not self.discrete:
            self.net = UNet()
        elif self.discrete:
            self.net = UNet(in_channel=1, out_channel=self.num_discrete)

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        # x_0 = batch
        t = np.random.randint(1, self.total_steps + 1)

        # For ddpm, NN predict added noise
        if self.diffusion_type == "ddpm":
            noise = torch.randn_like(batch)
            x_t = self.diffusion.forward_steps(batch, t, noise)
            pred_noise = self.net(x_t, t)
            loss = F.mse_loss(pred_noise, noise, reduction="sum") / np.prod(batch.size())
        
        # For sde, NN predict score function
        elif self.diffusion_type == "sde":
            _, g_t = self.diffusion.sde.drifts(torch.zeros_like(batch), t)
            mean, std = self.diffusion.sde.perturbation_kernel(batch, t)
            z = torch.randn_like(batch)

            x_t = mean + std * z  # x_t is perturbed x_0
            pred_score = self.net(x_t, t)
            target = -z / std   # Since desired derivative of the given kernel is - (x_t - mean) / std^2
            loss = g_t**2 * F.mse_loss(pred_score, target, reduction="sum") / np.prod(batch.size())

        # For discrete case, NN predict x_0
        elif self.diffusion_type == "discrete":
            x_t = self.diffusion.forward_steps(batch, t)
            pred_x_0_logit = self.net(x_t.float(), t)
            pred_x_0_prob = F.softmax(pred_x_0_logit, dim=1)
            loss = self.diffusion.loss(pred_x_0_prob, batch, x_t, t)

        self.log('train_loss', loss.item())

        return loss

    def generate(self, noise_batch):
        # noise batch = prior batch
        self.diffusion.sample(noise_batch, self.net)

    def validation_step(self, batch, batch_idx):
        self.generate(batch)

    def test_step(self, batch, batch_idx):
        self.generate(batch)
    

class DiffusionDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_name: str,
                 discrete: bool,
                 batch_size: int,
                 test_batch_size: int,
                 num_workers: int,
                 num_discrete: int = 2
                 ):
        super(DiffusionDataModule, self).__init__()
        self.data_name = data_name
        self.discrete = discrete
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.num_discrete = num_discrete

    def train_dataloader(self):
        return get_dataloader(self.data_name, self.discrete, self.batch_size, self.num_workers)
    
    def _noise_dataloader(self):
        if self.data_name in DATA_NAME and not self.discrete:
            noise = torch.randn(self.test_batch_size, 2)
        elif self.data_name == "mnist" and not self.discrete:
            noise = torch.randn(self.test_batch_size, 1, 32, 32)
        elif self.discrete:
            noise = torch.randint(0, self.num_discrete, (self.test_batch_size, 1, 32, 32))
        return DataLoader(noise, batch_size=self.test_batch_size, shuffle=False, num_workers=self.num_workers)

    def val_dataloader(self):
        return self._noise_dataloader()

    def test_dataloader(self):
        return self._noise_dataloader()