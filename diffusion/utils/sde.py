import numpy as np
import torch


class SDE():
    def __init__(self, total_steps, sde_type="VP", sde_info=None):
        self.total_steps = total_steps
        self.sde_type = sde_type
        assert sde_type in ["VP", "subVP", "VE"]
        self.sde_info = sde_info[sde_type]
        self.coef = self.compute_coef()

    def compute_coef(self):
        if self.sde_type == "VP" or self.sde_type == "subVP":
            beta_0 = self.sde_info["beta_min"]
            beta_1 = self.sde_info["beta_max"]
            return {"beta_0": beta_0, "beta_1": beta_1}

        if self.sde_type == "VE":
            sigma_min = self.sde_info["sigma_min"]
            sigma_max = self.sde_info["sigma_max"]
            return {"sigma_min": sigma_min, "sigma_max": sigma_max}
        
    def drifts(self, x_t, t : int):
        """
        Compute drifts f_t and g_t for SDE process for each timestep t
        """
        t = t / self.total_steps # t in [0, 1]
        if self.sde_type == "VP" or self.sde_type == "subVP":
            beta_t = self.coef["beta_0"] + t * (self.coef["beta_1"] - self.coef["beta_0"])
            f_t = - 0.5 * beta_t * x_t

            if self.sde_type == "VP":
                g_t = np.sqrt(beta_t)
                return f_t, g_t
            if self.sde_type == "subVP":
                discount = 1. - np.exp(-2 * self.coef["beta_0"] * t - (self.coef["beta_1"] - self.coef["beta_0"]) * t ** 2)
                g_t = np.sqrt(beta_t * discount)
                return f_t, g_t

        if self.sde_type == "VE":
            f_t = torch.zeros_like(x_t)
            sigma = self.coef["sigma_min"] * (self.coef["sigma_max"] / self.coef["sigma_min"]) ** t
            g_t = sigma * np.sqrt(2 * (np.log(self.coef["sigma_max"]) - np.log(self.coef["sigma_min"])))
            return f_t, g_t

    def perturbation_kernel(self, x_0, t : int):
        t = t / self.total_steps # t in [0, 1]
        if self.sde_type == "VP" or self.sde_type == "subVP":
            log_mean_coeff = -0.25 * t ** 2 * (self.coef["beta_1"] - self.coef["beta_0"]) - 0.5 * t * self.coef["beta_0"]
            mean = np.exp(log_mean_coeff) * x_0

            if self.sde_type == "VP":
                std = np.sqrt(1. - np.exp(2. * log_mean_coeff))
                return mean, std
            if self.sde_type == "subVP":
                std = 1. - np.exp(2. * log_mean_coeff)
                return mean, std

        if self.sde_type == "VE":
            mean = x_0
            std = self.coef["sigma_min"] * (self.coef["sigma_max"] / self.coef["sigma_min"]) ** t
            return mean, std
