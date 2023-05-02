import torch
import numpy as np
import os, sys
sys.path.append(os.getcwd()+"/Toy-Diffusion-Models")

from dataset import save2img, make_gif
from diffusion import DiffusionProcess


class DDPMProcess(DiffusionProcess):
    def __init__(self, **kwargs):
        super(DDPMProcess, self).__init__(**kwargs)
        assert self.discrete is False, "DDPM is only for continuous data"
        self.compute_coef(self.schedule)

    def forward_one_step(self, x_prev, t: int):
        """
        One step forward diffusion process computation from x_{t-1} to x_t
        Not used for actual training phase
        Range of t is [1, T], so for proper indexing of coefficient we need to subtract 1
        Check results with plot_forward_steps() function
        """
        return np.sqrt(self.coef["alphas"][t-1]) * x_prev + np.sqrt(self.coef["betas"][t-1]) * torch.randn_like(x_prev)

    def forward_steps(self, x_0, t: int, noise):
        """
        Forward diffusion process computation from x_0 to x_t
        """
        return np.sqrt(self.coef["alphas_cumprod"][t-1]) * x_0 + np.sqrt(1 - self.coef["alphas_cumprod"][t-1]) * noise

    def predict_x_0(self, x_t, t: int, pred_noise):
        return self.coef['pred_coef1'][t-1] * x_t - self.coef['pred_coef2'][t-1] * pred_noise

    def q_posterior(self, x_0, x_t, t):
        posterior_mean = self.coef['posterior_mean_coef1'][t-1] * x_0 + self.coef['posterior_mean_coef2'][t-1] * x_t
        posterior_log_variance_clipped = self.coef['posterior_log_variance_clipped'][t-1]
        return posterior_mean, posterior_log_variance_clipped

    @torch.no_grad()
    def backward_one_step(self, x_t, t: int, pred_noise, clip_denoised=True):
        """
        Backward diffusion process computation from x_t to x_{t-1}
        """
        # Approximate x_0 by NN
        pred_x_0 = self.predict_x_0(x_t, t, pred_noise)
        if clip_denoised and x_t.ndim > 2:
            pred_x_0.clamp_(-1., 1.)

        # Sample from q_posterior q(x_{t-1}|x_t, x_0)
        mean, log_variance = self.q_posterior(pred_x_0, x_t, t)
        noise = torch.randn_like(x_t)

        return mean + noise * np.exp((0.5 * log_variance))

    @torch.no_grad()
    def sample(self, noise, net):
        """
        Sample from backward diffusion process
        """
        x_t = noise
        data_type = "toy" if x_t.ndim == 2 else "img"

        save2img(x_t, self.result_dir + f"x_{self.total_steps}.png", data_type, self.discrete)
        trajectory = [self.result_dir + f"x_{self.total_steps}.png"]

        for t in reversed(range(1, self.total_steps+1)):
            pred_noise = net(x_t, t)
            x_t = self.backward_one_step(x_t, t, pred_noise)
            save2img(x_t, self.result_dir + f"x_{t-1}.png", data_type, self.discrete)
            trajectory.append(self.result_dir + f"x_{t-1}.png")

        save2img(x_t, f"generation.png", data_type, self.discrete)
        make_gif(trajectory, "generation")
        # [os.remove(path) for path in trajectory]


if __name__ == '__main__':
    from dataset import get_dataloader

    discrete = False
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    dataloader = get_dataloader(data_name="mnist", discrete=discrete, batch_size=16, num_workers=4)
    batch = next(iter(dataloader))

    diffusion = DDPMProcess(discrete=discrete, total_steps=50)
    diffusion.plot_forward_steps(batch.to(device))