import torch
from abc import ABC
import os
from dataset import save2img, make_gif
from diffusion.utils.coef import compute_ddpm_coef


class DiffusionProcess(ABC):
    def __init__(self, 
                 discrete: bool, 
                 total_steps: int, 
                 schedule: str = "cosine"):
        self.discrete = discrete
        self.total_steps = total_steps
        self.schedule = schedule
        self.result_dir = "./results/"
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

    def compute_coef(self, schedule):
        self.coef = compute_ddpm_coef(self.total_steps, schedule)

    def forward_one_step(self, x_prev, t):
        raise NotImplementedError

    def forward_steps(self, x_0, t, noise):
        raise NotImplementedError

    def plot_forward_steps(self, x_0):
        """
        Plot forward diffusion process computation from x_0 to x_T
        """

        data_type = "toy" if len(x_0.size()) == 2 else "img"
        save2img(x_0, self.result_dir + "x_0.png", data_type, self.discrete)
        save2img(torch.randn_like(x_0), "noise.png", data_type, self.discrete)
        trajectory = [self.result_dir + "x_0.png"]

        x_t = x_0
        for t in range(1, self.total_steps+1):
            x_t = self.forward_one_step(x_t, t)
            if x_t.ndim > 2:
                x_t.clamp_(-1., 1.)
            save2img(x_t, self.result_dir + f"x_{t}.png", data_type, self.discrete)
            trajectory.append(self.result_dir + f"x_{t}.png")
        
        make_gif(trajectory, "forward_process")
        [os.remove(path) for path in trajectory]