# Resemble Enhance

[![PyPI](https://img.shields.io/pypi/v/resemble-enhance.svg)](https://pypi.org/project/resemble-enhance/)
[![Hugging Face Space](https://img.shields.io/badge/Hugging%20Face%20%F0%9F%A4%97-Space-yellow)](https://huggingface.co/spaces/ResembleAI/resemble-enhance)
[![License](https://img.shields.io/github/license/resemble-ai/Resemble-Enhance.svg)](https://github.com/resemble-ai/resemble-enhance/blob/main/LICENSE)
[![Webpage](https://img.shields.io/badge/Webpage-Online-brightgreen)](https://www.resemble.ai/enhance/)

https://github.com/resemble-ai/resemble-enhance/assets/660224/bc3ec943-e795-4646-b119-cce327c810f1

Resemble Enhance is an AI-powered tool that aims to improve the overall quality of speech by performing denoising and enhancement. It consists of two modules: a denoiser, which separates speech from a noisy audio, and an enhancer, which further boosts the perceptual audio quality by restoring audio distortions and extending the audio bandwidth. The two models are trained on high-quality 44.1kHz speech data that guarantees the enhancement of your speech with high quality.

## Usage

### Installation

Install the stable version:

```bash
pip install resemble-enhance --upgrade
```

Or try the latest pre-release version:

```bash
pip install resemble-enhance --upgrade --pre
```

### Enhance

```
resemble-enhance in_dir out_dir
```

### Denoise only

```
resemble-enhance in_dir out_dir --denoise_only
```

### Web Demo

We provide a web demo built with Gradio, you can try it out [here](https://huggingface.co/spaces/ResembleAI/resemble-enhance), or also run it locally:

```
python app.py
```

## Train your own model

### Data Preparation

You need to prepare a foreground speech dataset and a background non-speech dataset. In addition, you can optionally provide the [WHAM! noise dataset](https://github.com/wham-team/wham) for extra noise augmentation and a RIR dataset ([examples](https://github.com/RoyJames/room-impulse-responses)).
The `scripts/download_data.py` script can automatically download several public
datasets and organize them. Run `python scripts/download_data.py --help` to
see all options. You can pass flags for specific datasets (e.g., `--vctk`,
`--dns-challenge`). If no flags are provided, all available datasets will be
downloaded.

The script will download and place the files into the following structure:

```bash
train_dataset
├── fg/en
│   └── ... .wav
├── bg/en
│   └── ... .wav
└── rir
    └── ... .wav
```

All audio files are downloaded as `.wav` files. Here is a breakdown of the downloaded datasets and where their files are placed:

- **Foreground (clean speech) files (`train_dataset/fg/en`):**
  - **DNS Challenge:** VocalSet and VCTK subsets.
  - **Voicebank:** Clean training set.
  - **LibriSpeech:** `train-clean-100` subset.
  - **DAPS:** The DAPS dataset.
  - **VCTK:** The VCTK corpus.
- **Background (noise) files (`train_dataset/bg/en`):**
  - **DNS Challenge:** Audioset and Freesound subsets.
  - **Voicebank:** Noisy training set (used here as background noise).
  - **RIR-NOISE:** Point-source noises.
- **Room Impulse Responses (`train_dataset/rir`):**
  - **DNS Challenge:** Impulse responses subset.
  - **RIR-NOISE:** Real RIRs from the Isotropic Noises subset.

To get started, first install the required packages. We recommend using a
virtual environment (e.g., conda or venv) to manage dependencies.

```bash
pip install -e .
```

Then, download the training data. This will download several speech and
noise datasets to the `train_dataset/` directory.

```bash
python scripts/download_data.py
```

The augmentation pipeline (RIR, reverb, WHAM! noise, etc.) can be controlled with flags such as `enable_rir`, `enable_reverb`, `enable_wham_noise`, and more in the YAML config files.

### Training

#### Denoiser Warmup

Though the denoiser is trained jointly with the enhancer, it is recommended for a warmup training first.

```bash
python -m resemble_enhance.denoiser.train --yaml config/denoiser.yaml runs/denoiser
```

#### Enhancer

Then, you can train the enhancer in two stages. The first stage is to train the autoencoder and vocoder. And the second stage is to train the latent conditional flow matching (CFM) model.

##### Stage 1

```bash
python -m resemble_enhance.enhancer.train --yaml config/enhancer_stage1.yaml runs/enhancer_stage1
```

##### Stage 2

```bash
python -m resemble_enhance.enhancer.train --yaml config/enhancer_stage2.yaml runs/enhancer_stage2
```

## Blog

Learn more on our [website](https://www.resemble.ai/enhance/)!