import json
import joblib
import numpy as np
import torch
import torch.nn as nn

from huggingface_hub import hf_hub_download
from diffusers import StableDiffusionPipeline
from sentence_transformers import SentenceTransformer

from .beta_ddim_scheduler import BetaDDIMScheduler
from .feature_builder import (
    build_combined_features,
    encode_prompt,
    init_latents,
    decode_latents,
    run_denoising_with_checkpoints,
)


class RegionMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.net(x)


def map_dataset_label_to_region(label_name):
    if label_name == "Structure":
        return "structure"
    if label_name == "Balanced":
        return "balanced"
    if label_name == "Texture":
        return "detail"
    raise ValueError(f"Unknown label: {label_name}")


def select_alpha_beta_from_grid(region, strength, alpha_beta_grid):
    candidates = [x for x in alpha_beta_grid if x["region"] == region]

    if region == "balanced":
        candidates = sorted(candidates, key=lambda x: x["alpha"])
        idx = min(len(candidates) - 1, int(strength * (len(candidates) - 1)))
        selected = candidates[idx]
        return selected["alpha"], selected["beta"]

    candidates = sorted(candidates, key=lambda x: abs(x["alpha"] - x["beta"]))
    idx = min(len(candidates) - 1, int(strength * (len(candidates) - 1)))
    selected = candidates[idx]
    return selected["alpha"], selected["beta"]


class AdaptiveBetaSampling:
    def __init__(
        self,
        hf_repo_id="ebad426623/adaptive-beta-sampling",
        sd_model_id="runwayml/stable-diffusion-v1-5",
        device=None,
        image_height=512,
        image_width=512,
    ):
        self.hf_repo_id = hf_repo_id
        self.sd_model_id = sd_model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.image_height = image_height
        self.image_width = image_width

        self._load_artifacts()
        self._load_models()

    def _load_artifacts(self):
        model_path = hf_hub_download(self.hf_repo_id, "model.pt")
        scaler_path = hf_hub_download(self.hf_repo_id, "scaler.joblib")
        label_map_path = hf_hub_download(self.hf_repo_id, "label_map.json")
        config_path = hf_hub_download(self.hf_repo_id, "config.json")
        grid_path = hf_hub_download(self.hf_repo_id, "alpha_beta_grid.json")

        self.scaler = joblib.load(scaler_path)

        with open(label_map_path, "r") as f:
            self.label_map = json.load(f)
        self.inv_label_map = {v: k for k, v in self.label_map.items()}

        with open(config_path, "r") as f:
            self.config = json.load(f)

        with open(grid_path, "r") as f:
            self.alpha_beta_grid = json.load(f)

        checkpoint = torch.load(model_path, map_location="cpu")
        self.model = RegionMLP(input_dim=checkpoint["input_dim"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def _load_models(self):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.sd_model_id,
            torch_dtype=self.dtype,
            safety_checker=None
        ).to(self.device)

        self.pipe.scheduler = BetaDDIMScheduler.from_config(self.pipe.scheduler.config)
        self.text_encoder_st = SentenceTransformer("all-MiniLM-L6-v2")

    def generate_image_with_beta(
        self,
        prompt_text,
        seed=42,
        num_steps=20,
        beta_alpha=1.0,
        beta_beta=1.0,
        guidance_scale=7.5,
        negative_prompt="",
    ):
        prompt_embeds = encode_prompt(self.pipe, prompt_text, device=self.device, negative_prompt=negative_prompt)
        latents = init_latents(
            self.pipe,
            device=self.device,
            dtype=self.dtype,
            height=self.image_height,
            width=self.image_width,
            seed=seed
        )

        final_latents, _ = run_denoising_with_checkpoints(
            pipe=self.pipe,
            prompt_embeds=prompt_embeds,
            latents=latents,
            device=self.device,
            num_steps=num_steps,
            checkpoint_indices=None,
            guidance_scale=guidance_scale,
            beta_alpha=beta_alpha,
            beta_beta=beta_beta,
        )

        return decode_latents(self.pipe, final_latents)

    def predict(self, prompt_text, seed=42, negative_prompt=""):
        cfg = self.config

        feat = build_combined_features(
            pipe=self.pipe,
            text_encoder_st=self.text_encoder_st,
            prompt_text=prompt_text,
            device=self.device,
            dtype=self.dtype,
            seed=seed,
            image_height=self.image_height,
            image_width=self.image_width,
            probe_steps=cfg["probe_steps"],
            probe_checkpoints=tuple(cfg["probe_checkpoints"]),
            default_alpha=cfg["default_alpha"],
            default_beta=cfg["default_beta"],
            guidance_scale=7.5,
            negative_prompt=negative_prompt,
        )

        combined_features = feat["combined_features"]
        x_scaled = self.scaler.transform(combined_features.reshape(1, -1))
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

        with torch.no_grad():
            logits = self.model(x_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_idx = int(torch.argmax(logits, dim=1).item())
            confidence = float(probs[0, pred_idx].item())

        pred_label = self.inv_label_map[pred_idx]
        pred_region = map_dataset_label_to_region(pred_label)

        strength = confidence
        alpha2, beta2 = select_alpha_beta_from_grid(pred_region, strength, self.alpha_beta_grid)

        return {
            "label": pred_label,
            "region": pred_region,
            "confidence": confidence,
            "alpha": alpha2,
            "beta": beta2,
            "features": feat,
        }

    def run(self, prompt_text, seed=42, negative_prompt=""):
        pred = self.predict(prompt_text, seed=seed, negative_prompt=negative_prompt)

        adaptive_img = self.generate_image_with_beta(
            prompt_text=prompt_text,
            seed=seed,
            num_steps=20,
            beta_alpha=pred["alpha"],
            beta_beta=pred["beta"],
            negative_prompt=negative_prompt,
        )

        return {
            **pred,
            "image": adaptive_img,
        }

    def compare(self, prompt_text, seed=42, negative_prompt=""):
        uniform_img = self.generate_image_with_beta(
            prompt_text=prompt_text,
            seed=seed,
            num_steps=20,
            beta_alpha=1.0,
            beta_beta=1.0,
            negative_prompt=negative_prompt,
        )

        fixed_img = self.generate_image_with_beta(
            prompt_text=prompt_text,
            seed=seed,
            num_steps=20,
            beta_alpha=self.config["default_alpha"],
            beta_beta=self.config["default_beta"],
            negative_prompt=negative_prompt,
        )

        adaptive_result = self.run(prompt_text, seed=seed, negative_prompt=negative_prompt)

        return {
            "uniform": uniform_img,
            "fixed": fixed_img,
            "adaptive": adaptive_result,
        }