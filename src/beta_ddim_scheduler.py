import numpy as np
import torch
from scipy.stats import beta as beta_dist
from diffusers import DDIMScheduler


class BetaDDIMScheduler(DDIMScheduler):
    def set_beta_params(self, beta_alpha=1.0, beta_beta=1.0):
        self.beta_alpha = float(beta_alpha)
        self.beta_beta = float(beta_beta)

    def _is_uniform_beta(self):
        a = getattr(self, "beta_alpha", 1.0)
        b = getattr(self, "beta_beta", 1.0)
        return abs(a - 1.0) < 1e-8 and abs(b - 1.0) < 1e-8

    def _beta_prev_timestep(self, timestep):
        if torch.is_tensor(timestep):
            timestep = int(timestep.item())
        else:
            timestep = int(timestep)

        ts = self.timesteps
        idx = (ts == timestep).nonzero(as_tuple=False)
        if len(idx) == 0:
            raise ValueError(f"Timestep {timestep} not found in self.timesteps")

        idx = int(idx[0].item())
        if idx == len(ts) - 1:
            return -1
        return int(ts[idx + 1].item())

    def set_timesteps(self, num_inference_steps, device=None):
        if self._is_uniform_beta():
            return super().set_timesteps(num_inference_steps, device=device)

        self.num_inference_steps = num_inference_steps

        max_t = self.config.num_train_timesteps - 1
        eps = 1e-4

        u = np.linspace(eps, 1.0 - eps, num_inference_steps)
        beta_u = beta_dist.ppf(u, self.beta_alpha, self.beta_beta)

        timesteps = np.round(beta_u * max_t).astype(np.int64)
        timesteps = np.clip(timesteps, 0, max_t)
        timesteps = np.sort(timesteps)[::-1]

        fixed = []
        prev = max_t + 1
        for t in timesteps:
            t = min(int(t), prev - 1)
            if t < 0:
                break
            fixed.append(t)
            prev = t

        if len(fixed) < num_inference_steps:
            fallback = np.linspace(max_t, 0, num_inference_steps).round().astype(np.int64)
            merged = sorted(set(fixed).union(set(fallback)), reverse=True)
            fixed = merged[:num_inference_steps]

        timesteps = np.array(fixed, dtype=np.int64)
        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.long)

    def step(self, *args, **kwargs):
        if self._is_uniform_beta():
            return DDIMScheduler.step(self, *args, **kwargs)

        model_output = kwargs.get("model_output", args[0] if len(args) > 0 else None)
        timestep = kwargs.get("timestep", args[1] if len(args) > 1 else None)
        sample = kwargs.get("sample", args[2] if len(args) > 2 else None)
        eta = kwargs.get("eta", 0.0)
        use_clipped_model_output = kwargs.get("use_clipped_model_output", False)
        generator = kwargs.get("generator", None)
        variance_noise = kwargs.get("variance_noise", None)
        return_dict = kwargs.get("return_dict", True)

        if self.num_inference_steps is None:
            raise ValueError("Run set_timesteps before calling step().")

        prev_timestep = self._beta_prev_timestep(timestep)

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t

        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
            pred_epsilon = model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** 0.5 * pred_original_sample) / beta_prod_t ** 0.5
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = alpha_prod_t ** 0.5 * sample - beta_prod_t ** 0.5 * model_output
            pred_epsilon = alpha_prod_t ** 0.5 * model_output + beta_prod_t ** 0.5 * sample
        else:
            raise ValueError(f"Unsupported prediction_type: {self.config.prediction_type}")

        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance ** 0.5

        if use_clipped_model_output:
            pred_epsilon = (sample - alpha_prod_t ** 0.5 * pred_original_sample) / beta_prod_t ** 0.5

        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** 0.5 * pred_epsilon
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction

        if eta > 0:
            from diffusers.utils.torch_utils import randn_tensor

            if variance_noise is not None and generator is not None:
                raise ValueError("Cannot pass both generator and variance_noise.")

            if variance_noise is None:
                variance_noise = randn_tensor(
                    model_output.shape,
                    generator=generator,
                    device=model_output.device,
                    dtype=model_output.dtype,
                )
            prev_sample = prev_sample + std_dev_t * variance_noise

        from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput

        if not return_dict:
            return (prev_sample, pred_original_sample)

        return DDIMSchedulerOutput(
            prev_sample=prev_sample,
            pred_original_sample=pred_original_sample,
        )