import torch
import numpy as np
import os, sys
sys.path.append(os.getcwd()+"/Toy-Diffusion-Models")

from dataset import save2img, make_gif
from diffusion.utils.sde import SDE
from diffusion import DiffusionProcess


class SDEProcess(DiffusionProcess):
    def __init__(self, sde_type="VP",
                 sde_info={"VP": {"beta_min": 0.1, "beta_max": 20},
                           "subVP": {"beta_min": 0.1, "beta_max": 20},
                           "VE": {"sigma_min": 0.01, "sigma_max": 50}}, 
                 **kwargs):
        super(SDEProcess, self).__init__(**kwargs)
        assert self.discrete is False, "DDPM is only for continuous data"
        self.dt = 1. / self.total_steps # step size
        self.sde = SDE(self.total_steps, sde_type, sde_info)

    def forward_one_step(self, x_prev, t):
        """
        Discretized forward SDE process for actual compuatation: 
        x_{t+1} = x_t + f_t(x_t) * dt + G_t * z_t * sqrt(dt)
        """
        f_t, g_t = self.sde.drifts(x_prev, t-1)
        z = torch.randn_like(x_prev)
        x_t = x_prev + f_t * self.dt + g_t * z * np.sqrt(self.dt)
        return x_t

    @torch.no_grad()
    def backward_one_step(self, x_t, t, pred_score, clip_denoised=True):
        """
        Discretized backward SDE process for actual compuatation:
        x_{t-1} = x_t - (f_t(x_t) - (G_t)^2 * pred_score) * dt + G_t * z_t * sqrt(dt)
        """
        z = torch.randn_like(x_t)
        f_t, g_t = self.sde.drifts(x_t, t)

        x_prev = x_t - (f_t - g_t**2 * pred_score) * self.dt + g_t * z * np.sqrt(self.dt)
        if clip_denoised and len(x_t.size()) > 2:
            x_prev.clamp_(-1., 1.)

        return x_prev

    @torch.no_grad()
    def sample(self, noise, net):
        """
        Sample from backward diffusion process
        """
        x_t = noise
        data_type = "toy" if len(x_t.size()) == 2 else "img"

        save2img(x_t, self.result_dir + f"x_{self.total_steps}.png", data_type, self.discrete)
        trajectory = [self.result_dir + f"x_{self.total_steps}.png"]

        for t in reversed(range(1, self.total_steps+1)):
            pred_score = net(x_t, t)
            x_t = self.backward_one_step(x_t, t, pred_score)
            save2img(x_t, self.result_dir + f"x_{t-1}.png", data_type, self.discrete)
            trajectory.append(self.result_dir + f"x_{t-1}.png")

        save2img(x_t, f"generation.png", data_type, self.discrete)
        make_gif(trajectory, "generation")


if __name__ == '__main__':
    from dataset import get_dataloader

    discrete = False
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    dataloader = get_dataloader(data_name="mnist", discrete=discrete, batch_size=16, num_workers=4)
    batch = next(iter(dataloader))

    diffusion = SDEProcess(discrete=discrete, total_steps=50)
    diffusion.plot_forward_steps(batch.to(device))