import logging
import random
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import librosa
import numpy as np
from scipy import signal

from ..utils import walk_paths
from .base import Effect

_logger = logging.getLogger(__name__)


@dataclass
class RandomRIR(Effect):
    rir_dir: Path | None
    rir_rate: int = 44_000
    rir_suffix: str = ".npy"
    deterministic: bool = False

    @cached_property
    def rir_paths(self):
        if self.rir_dir is None:
            return []
        return list(walk_paths(self.rir_dir, self.rir_suffix))

    def _sample_rir(self):
        if len(self.rir_paths) == 0:
            return None

        if self.deterministic:
            rir_path = self.rir_paths[0]
        else:
            rir_path = random.choice(self.rir_paths)

        rir = np.squeeze(np.load(rir_path))
        assert isinstance(rir, np.ndarray)

        return rir

    def apply(self, wav, sr):
        # ref: https://github.com/haoheliu/voicefixer_main/blob/b06e07c945ac1d309b8a57ddcd599ca376b98cd9/dataloaders/augmentation/magical_effects.py#L158

        if len(self.rir_paths) == 0:
            return wav

        length = len(wav)

        wav = librosa.resample(wav, orig_sr=sr, target_sr=self.rir_rate, res_type="kaiser_fast")
        rir = self._sample_rir()

        wav = signal.convolve(wav, rir, mode="same")

        actlev = np.max(np.abs(wav))
        if actlev > 0.99:
            wav = (wav / actlev) * 0.98

        wav = librosa.resample(wav, orig_sr=self.rir_rate, target_sr=sr, res_type="kaiser_fast")

        if abs(length - len(wav)) > 10:
            _logger.warning(f"length mismatch: {length} vs {len(wav)}")

        if length > len(wav):
            wav = np.pad(wav, (0, length - len(wav)))
        elif length < len(wav):
            wav = wav[:length]

        return wav


class RandomGaussianNoise(Effect):
    def __init__(self, alpha_range=(0.8, 1)):
        super().__init__()
        self.alpha_range = alpha_range

    def apply(self, wav, sr):
        noise = np.random.randn(*wav.shape)
        noise_energy = np.sum(noise**2)
        wav_energy = np.sum(wav**2)
        noise = noise * np.sqrt(wav_energy / noise_energy)
        alpha = random.uniform(*self.alpha_range)
        return wav * alpha + noise * (1 - alpha)


class RandomWhamNoise(Effect):
    def __init__(self, noise_dir: Path | None, deterministic: bool = False, alpha_range=(0.8, 1.0)):
        super().__init__()
        self.noise_dir = noise_dir
        self.alpha_range = alpha_range
        self.deterministic = deterministic

    @cached_property
    def noise_paths(self):
        if self.noise_dir is None:
            return []
        return list(walk_paths(self.noise_dir, ".wav"))

    def _sample_noise(self):
        if len(self.noise_paths) == 0:
            return None, None
        path = self.noise_paths[0] if self.deterministic else random.choice(self.noise_paths)
        wav, sr = librosa.load(path, sr=None, mono=True)
        return wav.astype(np.float32), sr

    def apply(self, wav, sr):
        if len(self.noise_paths) == 0:
            return wav
        noise, nsr = self._sample_noise()
        if noise is None:
            return wav
        if nsr != sr:
            noise = librosa.resample(noise, orig_sr=nsr, target_sr=sr, res_type="kaiser_fast")
        if len(noise) < len(wav):
            start = 0 if self.deterministic else random.randint(0, len(wav) - len(noise))
            padded = np.zeros(len(wav), dtype=np.float32)
            padded[start : start + len(noise)] = noise
            noise = padded
        elif len(noise) > len(wav):
            start = 0 if self.deterministic else random.randint(0, len(noise) - len(wav))
            noise = noise[start : start + len(wav)]
        noise_energy = np.sum(noise**2) + 1e-7
        wav_energy = np.sum(wav**2) + 1e-7
        noise = noise * np.sqrt(wav_energy / noise_energy)
        alpha = random.uniform(*self.alpha_range)
        return wav * alpha + noise * (1 - alpha)
