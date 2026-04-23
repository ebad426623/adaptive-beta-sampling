import numpy as np
import torch
from PIL import Image

from .fft_features import fft_band_energies, extract_fft_trend_features


@torch.no_grad()
def encode_prompt(pipe, prompt, device, negative_prompt=""):
    text_inputs = pipe.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    uncond_inputs = pipe.tokenizer(
        [negative_prompt],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids.to(device)
    uncond_input_ids = uncond_inputs.input_ids.to(device)

    text_embeddings = pipe.text_encoder(text_input_ids)[0]
    uncond_embeddings = pipe.text_encoder(uncond_input_ids)[0]

    return torch.cat([uncond_embeddings, text_embeddings], dim=0)


def init_latents(pipe, device, dtype, height=512, width=512, seed=42):
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(
        (1, pipe.unet.config.in_channels, height // 8, width // 8),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    latents = latents * pipe.scheduler.init_noise_sigma
    return latents


@torch.no_grad()
def decode_latents(pipe, latents):
    latents = latents / pipe.vae.config.scaling_factor
    image = pipe.vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image[0].detach().float().cpu().permute(1, 2, 0).numpy()
    image = (image * 255).round().astype("uint8")
    return Image.fromarray(image)


def set_scheduler_beta_params(pipe, alpha=1.0, beta=1.0):
    pipe.scheduler.set_beta_params(alpha, beta)


@torch.no_grad()
def run_denoising_with_checkpoints(
    pipe,
    prompt_embeds,
    latents,
    device,
    num_steps=20,
    checkpoint_indices=None,
    guidance_scale=7.5,
    beta_alpha=0.6,
    beta_beta=0.6,
):
    if checkpoint_indices is None:
        checkpoint_indices = set()
    else:
        checkpoint_indices = set(checkpoint_indices)

    set_scheduler_beta_params(pipe, beta_alpha, beta_beta)
    pipe.scheduler.set_timesteps(num_steps, device=device)

    saved_latents = {}
    current_latents = latents.clone()

    for step_idx, t in enumerate(pipe.scheduler.timesteps, start=1):
        latent_model_input = torch.cat([current_latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        noise_pred = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds
        ).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        step_output = pipe.scheduler.step(
            model_output=noise_pred,
            timestep=t,
            sample=current_latents,
            eta=0.0,
            return_dict=True
        )

        current_latents = step_output.prev_sample

        if step_idx in checkpoint_indices:
            saved_latents[step_idx] = current_latents.clone()

    return current_latents, saved_latents


def decode_checkpoint_latents(pipe, saved_latents_dict):
    return {step_idx: decode_latents(pipe, latents) for step_idx, latents in saved_latents_dict.items()}


def collect_fft_stats_from_images(images_by_step):
    return {step: fft_band_energies(img) for step, img in images_by_step.items()}


def get_prompt_embedding(text_encoder_st, prompt_text):
    emb = text_encoder_st.encode(
        prompt_text,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return emb.astype(np.float32)


def build_combined_features(
    pipe,
    text_encoder_st,
    prompt_text,
    device,
    dtype,
    seed=42,
    image_height=512,
    image_width=512,
    probe_steps=8,
    probe_checkpoints=(2, 4, 6, 8),
    default_alpha=0.6,
    default_beta=0.6,
    guidance_scale=7.5,
    negative_prompt="",
):
    text_features = get_prompt_embedding(text_encoder_st, prompt_text)
    prompt_embeds = encode_prompt(pipe, prompt_text, device=device, negative_prompt=negative_prompt)
    latents = init_latents(
        pipe,
        device=device,
        dtype=dtype,
        height=image_height,
        width=image_width,
        seed=seed
    )

    _, saved_latents = run_denoising_with_checkpoints(
        pipe=pipe,
        prompt_embeds=prompt_embeds,
        latents=latents,
        device=device,
        num_steps=probe_steps,
        checkpoint_indices=probe_checkpoints,
        guidance_scale=guidance_scale,
        beta_alpha=default_alpha,
        beta_beta=default_beta,
    )

    checkpoint_images = decode_checkpoint_latents(pipe, saved_latents)
    stats_by_step = collect_fft_stats_from_images(checkpoint_images)
    fft_features, aux = extract_fft_trend_features(stats_by_step)

    combined_features = np.concatenate([text_features, fft_features], axis=0).astype(np.float32)

    return {
        "text_features": text_features,
        "fft_features": fft_features,
        "combined_features": combined_features,
        "checkpoint_images": checkpoint_images,
        "stats_by_step": stats_by_step,
        "aux": aux,
    }