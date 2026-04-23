# Adaptive Beta Sampling

This repository provides inference code for Adaptive Beta Sampling, a diffusion scheduling approach that uses a short DDIM probe, FFT-based features, semantic prompt embeddings, and a trained classifier to select adaptive Beta parameters for image generation.

## Repository structure

- `notebooks/adaptive_beta_sampling_inference.ipynb`: inference notebook
- `src/beta_ddim_scheduler.py`: custom Beta-DDIM scheduler
- `src/fft_features.py`: FFT-based feature extraction
- `src/feature_builder.py`: prompt and probe feature construction
- `src/adaptive_sampling.py`: main adaptive inference pipeline

## Model artifacts

Trained artifacts are hosted on Hugging Face:

`ebad426623/adaptive-beta-sampling`

## Usage

Open the inference notebook and run all cells. The notebook will:
1. install dependencies
2. clone this repository
3. download trained artifacts from Hugging Face
4. run adaptive inference for a user prompt

## Method summary

Adaptive Beta Sampling works by:
1. running a short DDIM probe with a default Beta schedule
2. extracting FFT trend features from intermediate images
3. combining these with a semantic prompt embedding
4. predicting a region label
5. selecting a Beta pair from a predefined grid
6. generating the final image using the selected schedule