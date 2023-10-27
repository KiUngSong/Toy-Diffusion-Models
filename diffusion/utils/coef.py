import numpy as np


def compute_ddpm_coef(total_steps, schedule="linear"):
    betas = make_betas(total_steps, schedule)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
    variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "pred_coef1": np.sqrt(1.0 / alphas_cumprod),
        "pred_coef2": np.sqrt(1.0 / alphas_cumprod - 1),
        "variance": variance,
        "posterior_log_variance_clipped": np.log(np.maximum(variance, 1e-20)),
        "posterior_mean_coef1": betas
        * np.sqrt(alphas_cumprod_prev)
        / (1.0 - alphas_cumprod),
        "posterior_mean_coef2": (1.0 - alphas_cumprod_prev)
        * np.sqrt(alphas)
        / (1.0 - alphas_cumprod),
    }


def make_betas(total_steps, schedule="linear"):
    scale = 1000 / total_steps
    start, end = scale * 1e-4, scale * 2e-2
    if schedule == "linear":
        betas = np.linspace(start, end, total_steps)
    elif schedule == "cosine":
        betas = betas_for_alpha_bar(
            total_steps,
            lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(schedule)
    return betas


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)
