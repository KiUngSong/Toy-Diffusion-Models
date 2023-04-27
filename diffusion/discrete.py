import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import os, sys
sys.path.append(os.getcwd()+"/Toy-Diffusion-Models")

from dataset import save2img, make_gif
from diffusion import DiffusionProcess


class DiscreteProcess(DiffusionProcess):
    def __init__(self, num_discrete, **kwargs):
        super(DiscreteProcess, self).__init__(**kwargs)
        assert self.discrete is True, "Only designed for discrete data"
        self.num_discrete = num_discrete
        self.compute_coef(self.schedule)

    def to_prob_tensor(self, x):
        """
        Input: 
            x : Long tensor
        Output:
            x_prob : Float tensor
        """
        x_prob = F.one_hot(x.long(), self.num_discrete)[:,0,...]
        permute_order = (0, -1) + tuple(range(1, len(x.size()) - 1))
        x_prob = x_prob.permute(permute_order)
        return x_prob

    def forward_one_step(self, x_prev, t):
        """
        Input: 
            x_prev : Long tensor
            t : int
        Output:
            x_t : Long tensor
        """
        x_prev = F.one_hot(x_prev.long(), self.num_discrete)
        x_t_prob = (self.coef["alphas"][t-1] * x_prev + 
                    self.coef["betas"][t-1] * torch.ones_like(x_prev) / self.num_discrete)
        x_t = Categorical(x_t_prob).sample()
        return x_t
    
    def forward_steps(self, x_0, t):
        """
        Input: 
            x_prev : Long tensor
            t : int
        Output:
            x_t : Long tensor
        """
        x_0 = F.one_hot(x_0.long(), self.num_discrete)
        x_t_prob = (self.coef["alphas_cumprod"][t-1] * x_0 + 
                    (1 - self.coef["alphas_cumprod"][t-1]) * torch.ones_like(x_0) / self.num_discrete)
        x_t = Categorical(x_t_prob).sample()
        return x_t
    
    def q_posterior(self, x_0_prob, x_t, t):
        assert t >= 2, "Valid for timesteps t >= 2."
        x_t_prob = self.to_prob_tensor(x_t)
        assert x_0_prob.shape == x_t_prob.shape, "x_0_prob and x_t should have the same shape."

        # LHS and RHS of Eq. 13 in https://arxiv.org/abs/2102.05379 respectively
        prob1 = self.coef["alphas"][t-1] * x_t_prob + self.coef["betas"][t-1] * torch.ones_like(x_t_prob) / self.num_discrete
        prob2 = (self.coef["alphas_cumprod"][t-2] * x_0_prob + 
                 (1 - self.coef["alphas_cumprod"][t-2]) * torch.ones_like(x_0_prob) / self.num_discrete)
        posterior_prob = prob1 * prob2 / (prob1 * prob2).sum(dim=1, keepdim=True)

        return posterior_prob

    def loss(self, pred_x_0_prob, x_0, x_t, t):
        x_0_prob = self.to_prob_tensor(x_0)
        if t == 1:
            return F.cross_entropy(pred_x_0_prob, x_0_prob.float())
        else:
            pred_q_posterior = self.q_posterior(pred_x_0_prob, x_t, t)
            q_posterior = self.q_posterior(x_0_prob, x_t, t)
            loss = (q_posterior * (torch.log(q_posterior) - torch.log(pred_q_posterior))).sum(dim=1)
            return loss.mean()
        
    @torch.no_grad()
    def backward_one_step(self, x_t, t, pred_x_0_prob):
        """
        Backward diffusion process computation from x_t to x_{t-1}
        """
        if t >= 2:
            posterior_prob = self.q_posterior(pred_x_0_prob, x_t, t)
            posterior_prob = posterior_prob.permute(0, *tuple(range(2, len(posterior_prob.size()))), 1)
            posterior_prob = posterior_prob.unsqueeze(dim=1)
            x_prev = Categorical(posterior_prob).sample()
            return x_prev

        elif t == 1:
            return pred_x_0_prob.argmax(dim=1, keepdim=True)

    @torch.no_grad()
    def sample(self, noise, net):
        """
        Sample from backward diffusion process
        """
        x_t = noise

        save2img(x_t, self.result_dir + f"x_{self.total_steps}.png", discrete=self.discrete)
        trajectory = [self.result_dir + f"x_{self.total_steps}.png"]

        for t in reversed(range(1, self.total_steps+1)):
            pred_x_0_logit = net(x_t.float(), t)
            pred_x_0_prob = F.softmax(pred_x_0_logit, dim=1)
            x_t = self.backward_one_step(x_t, t, pred_x_0_prob)

            save2img(x_t, self.result_dir + f"x_{t-1}.png", discrete=self.discrete)
            trajectory.append(self.result_dir + f"x_{t-1}.png")

        save2img(x_t, f"generation.png", discrete=self.discrete)
        make_gif(trajectory, "generation")
        # [os.remove(path) for path in trajectory]


if __name__ == '__main__':
    from dataset import get_dataloader
    from model import UNet

    discrete = True
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    dataloader = get_dataloader(data_name="mnist", discrete=discrete, batch_size=16, num_workers=4)
    batch = next(iter(dataloader))
    
    diffusion = DiscreteProcess(discrete=discrete, num_discrete=2, total_steps=10)
    diffusion.plot_forward_steps(batch.to(device))