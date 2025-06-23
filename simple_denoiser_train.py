#!/usr/bin/env python3
"""
Training Script using Original Denoiser Model from resemble_enhance

Denoising Process:
1. Mix clean speech (foreground) with noise (background) using mix_fg_bg
2. Train model to recover clean speech from noisy mixture
3. Uses Raw Audio → Mix FG/BG → Model(STFT → UNet → iSTFT) → Raw Audio

The model IS learning to denoise! The mixing creates the noisy input.
Logs to Weights & Biases with comprehensive validation metrics.
"""

import argparse
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import wandb
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import librosa
import pesq
from pystoi import stoi
from scipy.signal import convolve

# Import the original model and utilities
from resemble_enhance.denoiser.denoiser import Denoiser
from resemble_enhance.denoiser.hparams import HParams as DenoiserHParams
from resemble_enhance.data.utils import mix_fg_bg
from resemble_enhance.hparams import HParams


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_denoiser_model(hp: HParams, device: torch.device, distort_prob: float = 0.0):
    """Create and configure the Denoiser model from resemble_enhance."""
    # Create denoiser-specific hyperparameters
    denoiser_hp = DenoiserHParams(
        batch_size_per_gpu=hp.batch_size_per_gpu,
        distort_prob=distort_prob,
        # Copy other relevant parameters from base HParams
        fg_dir=hp.fg_dir,
        bg_dir=hp.bg_dir,
        rir_dir=hp.rir_dir,
        wav_rate=hp.wav_rate,
        n_fft=hp.n_fft,
        win_size=hp.win_size,
        hop_size=hp.hop_size,
        num_mels=hp.num_mels,
        training_seconds=hp.training_seconds,
        nj=hp.nj,
        arch=hp.arch,
        use_ddp=False,  # We'll handle this separately
    )
    
    # Create the model
    model = Denoiser(denoiser_hp)
    return model.to(device)


class AudioDataset(Dataset):
    """Dataset for loading clean (foreground) and noise (background) audio following original approach."""
    
    def __init__(self, 
                 clean_dir: Path, 
                 noise_dir: Path,
                 sample_rate: int = 48000,
                 segment_length: float = 2.0,
                 training: bool = True,
                 distort_prob: float = 0.0,
                 max_samples: int = 1000,
                 rir_dir: Path = None):  # For reverb
        
        self.clean_dir = Path(clean_dir)
        self.noise_dir = Path(noise_dir)
        self.rir_dir = Path(rir_dir) if rir_dir else None
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_length * sample_rate)
        self.training = training
        self.distort_prob = distort_prob
        self.max_samples = max_samples
        
        # Get all audio files
        self.all_clean_files = list(self.clean_dir.rglob("*.wav"))
        self.noise_files = list(self.noise_dir.rglob("*.wav"))
        
        # Get RIR files if available
        self.rir_files = []
        if self.rir_dir and self.rir_dir.exists():
            self.rir_files = list(self.rir_dir.rglob("*.wav"))
            logger.info(f"Found {len(self.rir_files)} RIR files for reverb")
        
        if len(self.all_clean_files) == 0:
            raise ValueError(f"No audio files found in {clean_dir}")
        if len(self.noise_files) == 0:
            raise ValueError(f"No audio files found in {noise_dir}")
        
        logger.info(f"Found {len(self.all_clean_files)} clean files and {len(self.noise_files)} noise files")
        
        # Randomly sample subset for this epoch
        self.resample_epoch_data()
        
        # Show sampling info
        if len(self.all_clean_files) > self.max_samples:
            logger.info(f"Using {self.max_samples} random samples per epoch (out of {len(self.all_clean_files)} total)")
    
    def resample_epoch_data(self):
        """Randomly sample a subset of files for this epoch."""
        num_samples = min(self.max_samples, len(self.all_clean_files))
        self.clean_files = random.sample(self.all_clean_files, num_samples)
        logger.info(f"Sampled {len(self.clean_files)} files for this epoch")
    
    def load_audio(self, file_path: Path) -> np.ndarray:
        """Load and preprocess audio file."""
        try:
            # Load audio - use backend to avoid deprecation warning
            import soundfile as sf
            audio, sr = sf.read(file_path, dtype='float32')
            
            # Convert to tensor format expected by torchaudio
            if audio.ndim == 1:
                waveform = torch.from_numpy(audio).unsqueeze(0)
            else:
                waveform = torch.from_numpy(audio).T  # soundfile uses (samples, channels)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Always resample to ensure consistent 48kHz
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to numpy
            audio = waveform.squeeze().numpy()
            
            # Normalize
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            return audio
            
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")
            # Return silence if loading fails
            return np.zeros(self.segment_samples)
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range."""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio
    
    def __len__(self):
        return len(self.clean_files)
    
    def apply_reverb(self, audio: np.ndarray, rir: np.ndarray) -> np.ndarray:
        """Apply reverb using RIR convolution."""
        # Normalize RIR
        rir = rir / np.max(np.abs(rir) + 1e-10)
        
        # Convolve audio with RIR
        reverbed = convolve(audio, rir, mode='same')
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(reverbed))
        if max_val > 0:
            reverbed = reverbed / max_val
        
        return reverbed
    
    def create_robust_noise(self, target_length: int) -> np.ndarray:
        """Create robust noise by mixing multiple noise sources."""
        # Determine number of noise sources to mix (2-4 for training, 1 for val)
        if self.training:
            num_sources = random.randint(2, 4)
        else:
            num_sources = 1
        
        # Initialize combined noise
        combined_noise = np.zeros(target_length)
        
        for i in range(num_sources):
            # Load a random noise file
            noise_file = random.choice(self.noise_files)
            noise_audio = self.load_audio(noise_file)
            
            # Match length
            if len(noise_audio) < target_length:
                repeats = int(np.ceil(target_length / len(noise_audio)))
                noise_audio = np.tile(noise_audio, repeats)
            
            # Random crop
            if len(noise_audio) > target_length:
                start_idx = random.randint(0, len(noise_audio) - target_length)
                noise_audio = noise_audio[start_idx:start_idx + target_length]
            
            # Random gain for this noise source (more aggressive range)
            gain = random.uniform(0.5, 1.5) if self.training else 1.0
            
            # Add to combined noise
            combined_noise += gain * noise_audio
        
        # Normalize combined noise
        combined_noise = self.normalize_audio(combined_noise)
        
        return combined_noise
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        # Load foreground (clean) audio
        fg_file = self.clean_files[idx]
        fg_audio = self.load_audio(fg_file)
        
        # Random crop if training
        if self.training and len(fg_audio) > self.segment_samples:
            start_idx = random.randint(0, len(fg_audio) - self.segment_samples)
            fg_audio = fg_audio[start_idx:start_idx + self.segment_samples]
        elif len(fg_audio) < self.segment_samples:
            # Pad if too short
            padding = self.segment_samples - len(fg_audio)
            fg_audio = np.pad(fg_audio, (0, padding), mode='constant')
        else:
            fg_audio = fg_audio[:self.segment_samples]
        
        # Normalize clean audio
        fg_audio = self.normalize_audio(fg_audio)
        
        # Create distorted version of foreground (fg_dwav)
        fg_dwav = fg_audio.copy()
        
        # Apply reverb to fg_dwav with probability
        if self.training and self.rir_files and random.random() < 0.3:
            rir_file = random.choice(self.rir_files)
            rir = self.load_audio(rir_file)
            # Use only first part of RIR to avoid too much reverb
            rir = rir[:min(len(rir), self.sample_rate // 2)]  # Max 0.5 second reverb
            fg_dwav = self.apply_reverb(fg_dwav, rir)
        
        # Create robust background noise (potentially multiple sources)
        bg_audio = self.create_robust_noise(len(fg_audio))
        
        # Create distorted version of background (bg_dwav)
        bg_dwav = bg_audio.copy()
        
        # Apply reverb to bg_dwav with probability (different RIR)
        if self.training and self.rir_files and random.random() < 0.2:
            rir_file = random.choice(self.rir_files)
            rir = self.load_audio(rir_file)
            rir = rir[:min(len(rir), self.sample_rate // 4)]  # Shorter reverb for noise
            bg_dwav = self.apply_reverb(bg_dwav, rir)
        
        # Additional distortions with probability
        if self.training and random.random() < self.distort_prob:
            # Frequency distortion (simple high-pass filter effect)
            if random.random() < 0.3:
                # Remove low frequencies to simulate phone/radio quality
                cutoff = random.uniform(200, 500)  # Hz
                b, a = self._butter_highpass(cutoff, self.sample_rate, order=3)
                from scipy.signal import filtfilt
                fg_dwav = filtfilt(b, a, fg_dwav)
            
            # Clipping distortion
            if random.random() < 0.2:
                clip_level = random.uniform(0.5, 0.8)
                fg_dwav = np.clip(fg_dwav, -clip_level, clip_level)
                fg_dwav = self.normalize_audio(fg_dwav)
        
        # Return in format expected by original model
        return {
            'fg_wav': torch.FloatTensor(fg_audio),   # Clean target
            'bg_wav': torch.FloatTensor(bg_audio),   # Background noise (clean)
            'fg_dwav': torch.FloatTensor(fg_dwav),   # Distorted foreground
            'bg_dwav': torch.FloatTensor(bg_dwav),   # Distorted background
        }
    
    def _butter_highpass(self, cutoff, fs, order=5):
        """Design a butterworth highpass filter."""
        from scipy.signal import butter
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function following original data format."""
    # Stack all tensors
    return {
        'fg_wavs': torch.stack([item['fg_wav'] for item in batch]),
        'bg_wavs': torch.stack([item['bg_wav'] for item in batch]),
        'fg_dwavs': torch.stack([item['fg_dwav'] for item in batch]),
        'bg_dwavs': torch.stack([item['bg_dwav'] for item in batch]),
    }


def calculate_metrics(clean_audio: np.ndarray, 
                     enhanced_audio: np.ndarray, 
                     noisy_audio: np.ndarray,
                     sample_rate: int) -> Dict[str, float]:
    """Calculate various audio quality metrics."""
    metrics = {}
    
    try:
        # Ensure all audio has same length
        min_len = min(len(clean_audio), len(enhanced_audio), len(noisy_audio))
        clean_audio = clean_audio[:min_len]
        enhanced_audio = enhanced_audio[:min_len]
        noisy_audio = noisy_audio[:min_len]
        
        # PESQ (Perceptual Evaluation of Speech Quality) - downsample to 16kHz for PESQ
        try:
            # Downsample to 16kHz for PESQ calculation only
            target_sr = 16000
            if sample_rate != target_sr:
                clean_16k = librosa.resample(clean_audio, orig_sr=sample_rate, target_sr=target_sr)
                enhanced_16k = librosa.resample(enhanced_audio, orig_sr=sample_rate, target_sr=target_sr)
                pesq_score = pesq.pesq(target_sr, clean_16k, enhanced_16k, 'wb')
            else:
                pesq_score = pesq.pesq(sample_rate, clean_audio, enhanced_audio, 'wb')
            metrics['pesq'] = pesq_score
        except Exception as e:
            logger.warning(f"PESQ calculation failed: {e}")
            metrics['pesq'] = 0.0
        
        # ESTOI (Extended STOI) - PRIMARY METRIC for speech intelligibility at 48kHz
        # Target: ≥ 0.85 - Works natively at 48k; better than STOI for wide-band speech
        try:
            estoi_score = stoi(clean_audio, enhanced_audio, sample_rate, extended=True)
            metrics['estoi'] = estoi_score
            
            # Regular STOI (for comparison)
            stoi_score = stoi(clean_audio, enhanced_audio, sample_rate, extended=False)
            metrics['stoi'] = stoi_score
        except Exception as e:
            logger.warning(f"STOI/ESTOI calculation failed: {e}")
            metrics['stoi'] = 0.0
            metrics['estoi'] = 0.0
        
        # SNR improvement
        try:
            clean_power = np.mean(clean_audio ** 2) + 1e-10
            noise_power_before = np.mean((noisy_audio - clean_audio) ** 2) + 1e-10
            noise_power_after = np.mean((enhanced_audio - clean_audio) ** 2) + 1e-10
            
            snr_before = 10 * np.log10(clean_power / noise_power_before)
            snr_after = 10 * np.log10(clean_power / noise_power_after)
            
            metrics['snr_improvement'] = snr_after - snr_before
            metrics['snr_before'] = snr_before
            metrics['snr_after'] = snr_after
        except Exception as e:
            logger.warning(f"SNR calculation failed: {e}")
            metrics['snr_improvement'] = 0.0
            metrics['snr_before'] = 0.0
            metrics['snr_after'] = 0.0
        
        # SPECTRAL FIDELITY METRICS - PRIMARY for "crispness" and high-freq accuracy
        try:
            # Compute spectrograms
            clean_spec = np.abs(librosa.stft(clean_audio, n_fft=2048, hop_length=512))
            enhanced_spec = np.abs(librosa.stft(enhanced_audio, n_fft=2048, hop_length=512))
            
            # Ensure same shape
            min_frames = min(clean_spec.shape[1], enhanced_spec.shape[1])
            clean_spec = clean_spec[:, :min_frames]
            enhanced_spec = enhanced_spec[:, :min_frames]
            
            # Log-Spectral Distance (LSD) - PRIMARY METRIC for spectral fidelity
            # Target: < 1 dB - Captures high-freq errors the ear notices most
            lsd = np.mean(np.sqrt(np.mean((20 * np.log10(enhanced_spec + 1e-10) - 
                                        20 * np.log10(clean_spec + 1e-10))**2, axis=0)))
            metrics['lsd'] = lsd
            
            # Spectral Convergence - Complement to LSD
            # Target: < 0.3 - Lower is better
            spectral_convergence = np.linalg.norm(enhanced_spec - clean_spec, 'fro') / np.linalg.norm(clean_spec, 'fro')
            metrics['spectral_convergence'] = spectral_convergence
            
        except Exception as e:
            logger.warning(f"Spectral metrics calculation failed: {e}")
            metrics['lsd'] = 0.0
            metrics['spectral_convergence'] = 0.0
        
        # SI-SDR - PRIMARY METRIC for noise-removal strength & artifacts
        # Target: > 10 dB - Modern replacement for SNR
        try:
            def si_sdr(reference, estimation):
                """Calculate SI-SDR - Scale-Invariant Signal-to-Distortion Ratio"""
                eps = np.finfo(reference.dtype).eps
                reference = reference.reshape(-1)
                estimation = estimation.reshape(-1)
                
                # Zero-mean signals
                reference = reference - np.mean(reference)
                estimation = estimation - np.mean(estimation)
                
                # SI-SDR calculation
                reference_energy = np.sum(reference ** 2) + eps
                optimal_scaling = np.sum(reference * estimation) / reference_energy
                projection = optimal_scaling * reference
                noise = estimation - projection
                
                si_sdr_value = 10 * np.log10(np.sum(projection ** 2) / (np.sum(noise ** 2) + eps))
                return si_sdr_value
            
            si_sdr_enhanced = si_sdr(clean_audio, enhanced_audio)
            si_sdr_noisy = si_sdr(clean_audio, noisy_audio)
            si_sdr_improvement = si_sdr_enhanced - si_sdr_noisy
            
            metrics['si_sdr'] = si_sdr_enhanced
            metrics['si_sdr_improvement'] = si_sdr_improvement
            
        except Exception as e:
            logger.warning(f"SI-SDR calculation failed: {e}")
            metrics['si_sdr'] = 0.0
            metrics['si_sdr_improvement'] = 0.0
        
    except Exception as e:
        logger.warning(f"Error calculating metrics: {e}")
        metrics = {
            'pesq': 0.0, 'stoi': 0.0, 'estoi': 0.0, 'snr_improvement': 0.0,
            'lsd': 0.0, 'spectral_convergence': 0.0, 'si_sdr': 0.0, 'si_sdr_improvement': 0.0
        }
    
    return metrics


class MultiScaleSTFTLoss(nn.Module):
    """Multi-scale STFT loss for better spectral reconstruction."""
    
    def __init__(self, fft_sizes=[512, 1024, 2048], hop_ratios=[0.25, 0.25, 0.25], win_ratios=[1.0, 1.0, 1.0]):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_lengths = [int(fft_size * hop_ratio) for fft_size, hop_ratio in zip(fft_sizes, hop_ratios)]
        self.win_lengths = [int(fft_size * win_ratio) for fft_size, win_ratio in zip(fft_sizes, win_ratios)]
        
    def stft(self, x, fft_size, hop_length, win_length):
        """Compute STFT."""
        return torch.stft(
            x, n_fft=fft_size, hop_length=hop_length, win_length=win_length,
            window=torch.hann_window(win_length).to(x.device),
            return_complex=True
        )
    
    def spectral_loss(self, pred_stft, target_stft):
        """Compute spectral convergence and log magnitude loss."""
        pred_mag = torch.abs(pred_stft)
        target_mag = torch.abs(target_stft)
        
        # Spectral convergence
        spectral_conv = torch.norm(pred_mag - target_mag, p='fro') / torch.norm(target_mag, p='fro')
        
        # Log magnitude loss
        log_mag_loss = F.l1_loss(torch.log(pred_mag + 1e-7), torch.log(target_mag + 1e-7))
        
        return spectral_conv + log_mag_loss
    
    def forward(self, pred, target):
        """Calculate multi-scale STFT loss."""
        total_loss = 0.0
        
        for fft_size, hop_length, win_length in zip(self.fft_sizes, self.hop_lengths, self.win_lengths):
            pred_stft = self.stft(pred, fft_size, hop_length, win_length)
            target_stft = self.stft(target, fft_size, hop_length, win_length)
            
            stft_loss = self.spectral_loss(pred_stft, target_stft)
            total_loss += stft_loss
        
        return total_loss / len(self.fft_sizes)


def train_epoch(model: nn.Module, 
                dataloader: DataLoader, 
                optimizer: torch.optim.Optimizer,
                hp: HParams,
                device: torch.device,
                epoch: int) -> Dict[str, float]:
    """Train for one epoch using original denoiser approach."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        for key in batch:
            batch[key] = batch[key].to(device)
        
        # ROBUST DENOISING HAPPENS HERE:
        # 1. We mix clean speech (fg) with challenging noise (bg) 
        # 2. Use distorted versions for more robust training
        # 3. Model learns to recover clean speech from very noisy mixture
        
        # Use more aggressive mixing ratios for challenging conditions
        # Original range is typically [0.1, 0.5], we'll use [0.3, 1.0] for harder cases
        if hp.mix_alpha_range:
            min_alpha, max_alpha = hp.mix_alpha_range
            # Make noise louder for more challenging training
            alpha = random.uniform(min_alpha * 3, min(max_alpha * 2, 1.0))
        else:
            alpha = random.uniform(0.3, 1.0)  # Default aggressive range
        
        # Randomly decide whether to use distorted versions (more challenging)
        use_distorted = random.random() < 0.7  # 70% of the time use distorted
        
        if use_distorted:
            fg_wavs = batch['fg_dwavs']  # Distorted speech (with reverb, filtering, etc.)
            bg_wavs = batch['bg_dwavs']  # Distorted noise (multi-source, reverb)
        else:
            fg_wavs = batch['fg_wavs']   # Clean speech
            bg_wavs = batch['bg_wavs']   # Clean noise
        
        # Always use clean target for loss calculation
        target_wavs = batch['fg_wavs']
        
        # CREATE VERY NOISY AUDIO by mixing speech + challenging noise
        mixed_wavs = mix_fg_bg(fg_wavs, bg_wavs, alpha=alpha)
        
        # DENOISING: Model learns to extract clean speech from very noisy mixture
        optimizer.zero_grad()
        enhanced_wavs = model(mixed_wavs, target_wavs)  # Input: noisy, Target: clean
        
        # Get loss from model (it computes loss internally)
        if hasattr(model, 'losses'):
            loss = sum(model.losses.values())
        else:
            # Fallback to L1 loss if model doesn't compute loss
            loss = F.l1_loss(enhanced_wavs, target_wavs)
        
        # Backward pass
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
        
        # Log to wandb every 100 steps
        if batch_idx % 100 == 0:
            step = epoch * len(dataloader) + batch_idx
            wandb.log({
                'train_loss': loss.item(),
                'learning_rate': optimizer.param_groups[0]['lr'],
                'epoch': epoch,
                'step': step
            })
    
    avg_loss = total_loss / num_batches
    return {'train_loss': avg_loss}


def validate_epoch(model: nn.Module, 
                   dataloader: DataLoader, 
                   hp: HParams,
                   device: torch.device,
                   epoch: int,
                   sample_rate: int) -> Dict[str, float]:
    """Validate for one epoch using original denoiser approach."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    # Metrics accumulators
    pesq_scores = []
    stoi_scores = []
    estoi_scores = []
    snr_improvements = []
    lsd_scores = []
    spectral_convergence_scores = []
    si_sdr_scores = []
    si_sdr_improvements = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Validation {epoch}")
        for batch in pbar:
            # Move batch to device
            for key in batch:
                batch[key] = batch[key].to(device)
            
            # Use fixed alpha for validation for consistent evaluation
            alpha = 0.5  # 50% mix ratio for validation
            fg_wavs = batch['fg_wavs']  # Clean speech
            bg_wavs = batch['bg_wavs']  # Noise
            
            # Create noisy audio by mixing speech with noise
            mixed_wavs = mix_fg_bg(fg_wavs, bg_wavs, alpha=alpha)
            
            # Forward pass through model
            enhanced_wavs = model(mixed_wavs, fg_wavs)
            
            # Calculate loss
            if hasattr(model, 'losses'):
                loss = sum(model.losses.values())
            else:
                loss = F.l1_loss(enhanced_wavs, fg_wavs)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Get audio for metrics (first sample in batch)
            clean_audio = fg_wavs[0].cpu().numpy()
            noisy_audio = mixed_wavs[0].cpu().numpy()
            enhanced_audio = enhanced_wavs[0].cpu().numpy()
            
            # Calculate metrics
            metrics = calculate_metrics(clean_audio, enhanced_audio, noisy_audio, sample_rate)
            
            # Collect all metrics
            if 'pesq' in metrics and metrics['pesq'] > 0:
                pesq_scores.append(metrics['pesq'])
            if 'stoi' in metrics and metrics['stoi'] > 0:
                stoi_scores.append(metrics['stoi'])
            if 'estoi' in metrics and metrics['estoi'] > 0:
                estoi_scores.append(metrics['estoi'])
            if 'snr_improvement' in metrics:
                snr_improvements.append(metrics['snr_improvement'])
            if 'lsd' in metrics and metrics['lsd'] > 0:
                lsd_scores.append(metrics['lsd'])
            if 'spectral_convergence' in metrics and metrics['spectral_convergence'] > 0:
                spectral_convergence_scores.append(metrics['spectral_convergence'])
            if 'si_sdr' in metrics:
                si_sdr_scores.append(metrics['si_sdr'])
            if 'si_sdr_improvement' in metrics:
                si_sdr_improvements.append(metrics['si_sdr_improvement'])
    
    # Calculate average metrics
    avg_loss = total_loss / num_batches
    avg_pesq = np.mean(pesq_scores) if pesq_scores else 0.0
    avg_stoi = np.mean(stoi_scores) if stoi_scores else 0.0
    avg_estoi = np.mean(estoi_scores) if estoi_scores else 0.0
    avg_snr_improvement = np.mean(snr_improvements) if snr_improvements else 0.0
    avg_lsd = np.mean(lsd_scores) if lsd_scores else 0.0
    avg_spectral_convergence = np.mean(spectral_convergence_scores) if spectral_convergence_scores else 0.0
    avg_si_sdr = np.mean(si_sdr_scores) if si_sdr_scores else 0.0
    avg_si_sdr_improvement = np.mean(si_sdr_improvements) if si_sdr_improvements else 0.0
    
    return {
        'val_loss': avg_loss,
        'val_pesq': avg_pesq,
        'val_stoi': avg_stoi,
        'val_estoi': avg_estoi,
        'val_snr_improvement': avg_snr_improvement,
        'val_lsd': avg_lsd,
        'val_spectral_convergence': avg_spectral_convergence,
        'val_si_sdr': avg_si_sdr,
        'val_si_sdr_improvement': avg_si_sdr_improvement
    }


def main():
    parser = argparse.ArgumentParser(description='Train Simple CNN Denoiser')
    parser.add_argument('--clean_dir', type=Path, default='train_dataset/fg/en',
                        help='Directory containing clean speech files')
    parser.add_argument('--noise_dir', type=Path, default='train_dataset/bg/en',
                        help='Directory containing noise files')
    parser.add_argument('--rir_dir', type=Path, default='train_dataset/rir',
                        help='Directory containing RIR files for reverb (optional)')
    parser.add_argument('--output_dir', type=Path, default='runs/simple_denoiser',
                        help='Output directory for model checkpoints')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--sample_rate', type=int, default=48000, help='Audio sample rate')
    parser.add_argument('--segment_length', type=float, default=2.0, help='Audio segment length in seconds')
    parser.add_argument('--arch', type=str, default='unet', choices=['unet', 'convtasnet'], 
                        help='Model architecture')
    parser.add_argument('--max_samples', type=int, default=1000, 
                        help='Maximum samples per epoch for faster training')
    parser.add_argument('--distort_prob', type=float, default=0.3,
                        help='Probability of applying distortions (reverb, filtering, clipping)')
    parser.add_argument('--wandb_project', type=str, default='simple-denoiser', help='W&B project name')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb with comprehensive config
    wandb.init(
        project=args.wandb_project,
        config={
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'learning_rate': args.lr,
            'sample_rate': args.sample_rate,
            'segment_length': args.segment_length,
            'max_samples_per_epoch': args.max_samples,
            'architecture': args.arch,
            'model': 'Original Denoiser from resemble_enhance',
            'hop_size': 420,
            'n_fft': 2048,
            'optimizer': 'Adam',
            'scheduler': 'OneCycleLR',
            'loss_function': 'L1 (internal)',
            'data_approach': 'Raw Audio → Mix FG/BG → Model(STFT → UNet → iSTFT) → Raw Audio',
            'denoising_task': 'Speech + Multi-Noise → Speech (aggressive alpha: 0.3-1.0)',
            'distortions': f'distort_prob={args.distort_prob} (reverb, filtering, clipping)',
            'robust_noise': 'Multi-source mixing (2-4 sources), reverb, aggressive gains',
            'primary_metrics': 'ESTOI, SI-SDR, LSD, Spectral_Convergence'
        }
    )
    
    # Define W&B metrics for proper graphing
    wandb.define_metric("epoch")
    wandb.define_metric("step")
    
    # Training metrics
    wandb.define_metric("train_loss", step_metric="step")
    wandb.define_metric("train_loss_epoch", step_metric="epoch")
    wandb.define_metric("learning_rate", step_metric="step")
    
    # Validation metrics - all vs epoch
    wandb.define_metric("val_loss", step_metric="epoch")
    wandb.define_metric("val_pesq", step_metric="epoch")
    wandb.define_metric("val_stoi", step_metric="epoch")
    wandb.define_metric("val_estoi", step_metric="epoch")
    wandb.define_metric("val_snr_improvement", step_metric="epoch")
    wandb.define_metric("val_lsd", step_metric="epoch")
    wandb.define_metric("val_spectral_convergence", step_metric="epoch")
    wandb.define_metric("val_si_sdr", step_metric="epoch")
    wandb.define_metric("val_si_sdr_improvement", step_metric="epoch")
    
    # Set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Create datasets
    train_dataset = AudioDataset(
        clean_dir=args.clean_dir,
        noise_dir=args.noise_dir,
        sample_rate=args.sample_rate,
        segment_length=args.segment_length,
        training=True,
        max_samples=args.max_samples,
        distort_prob=args.distort_prob,
        rir_dir=args.rir_dir if args.rir_dir and args.rir_dir.exists() else None
    )
    
    val_dataset = AudioDataset(
        clean_dir=args.clean_dir,
        noise_dir=args.noise_dir,
        sample_rate=args.sample_rate,
        segment_length=args.segment_length,
        training=False,
        max_samples=min(100, args.max_samples),  # Smaller validation set
        distort_prob=0.0,  # No distortions for validation
        rir_dir=None  # No reverb for validation
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Create HParams for the model
    hp = HParams(
        fg_dir=Path(args.clean_dir),
        bg_dir=Path(args.noise_dir),
        rir_dir=Path(args.rir_dir) if args.rir_dir and args.rir_dir.exists() else None,
        wav_rate=args.sample_rate,
        training_seconds=args.segment_length,
        batch_size_per_gpu=args.batch_size,
        mix_alpha_range=(0.2, 0.8),  # Model's internal range (we override in training)
        arch=args.arch,
        hop_size=420,  # Default from original code
        n_fft=2048,
        win_size=2048,
        num_mels=128,
        nj=4,
    )
    
    # Create model using original Denoiser with our robust noising pipeline:
    # 1. Multi-source noise mixing (2-4 sources combined)
    # 2. Reverb via RIR convolution when available
    # 3. Additional distortions (filtering, clipping) with distort_prob
    # 4. Aggressive noise levels (alpha 0.3-1.0 instead of 0.1-0.5)
    # All these make training more challenging -> more robust model!
    
    # Create model using original Denoiser
    model = create_denoiser_model(hp, device, args.distort_prob)
    
    # Create optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader)
    )
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Training batches per epoch: {len(train_loader)} (with max_samples={args.max_samples})")
    logger.info(f"Validation batches: {len(val_loader)}")
    
    wandb.log({
        'total_params': total_params,
        'trainable_params': trainable_params
    })
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        
        # Resample training data for each epoch to get different subset
        train_dataset.resample_epoch_data()
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, hp, device, epoch)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, hp, device, epoch, args.sample_rate)
        
        # Update scheduler
        scheduler.step()
        
        # Log epoch-level metrics to wandb
        epoch_metrics = {
            **train_metrics,
            **val_metrics,
            'epoch': epoch,
            'train_loss_epoch': train_metrics['train_loss']
        }
        wandb.log(epoch_metrics)
        
        # Display PRIMARY METRICS with targets
        logger.info(f"=== Epoch {epoch + 1}/{args.epochs} Results ===")
        logger.info(f"Train Loss: {train_metrics['train_loss']:.4f}")
        logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}")
        
        # 🎯 PRIMARY METRICS (with targets)
        logger.info("🎯 PRIMARY METRICS:")
        
        # ESTOI - Speech intelligibility (Target: ≥ 0.85)
        if val_metrics['val_estoi'] > 0:
            status = "✅ PASS" if val_metrics['val_estoi'] >= 0.85 else "❌ FAIL"
            logger.info(f"  ESTOI: {val_metrics['val_estoi']:.3f} (≥0.85) {status}")
        
        # SI-SDR - Noise removal (Target: > 10 dB)
        if val_metrics['val_si_sdr'] != 0:
            status = "✅ PASS" if val_metrics['val_si_sdr'] > 10.0 else "❌ FAIL"
            logger.info(f"  SI-SDR: {val_metrics['val_si_sdr']:.2f} dB (>10) {status}")
        
        # LSD - Spectral fidelity (Target: < 1 dB)
        if val_metrics['val_lsd'] > 0:
            status = "✅ PASS" if val_metrics['val_lsd'] < 1.0 else "❌ FAIL"
            logger.info(f"  LSD: {val_metrics['val_lsd']:.3f} dB (<1.0) {status}")
        
        # Spectral Convergence (Target: < 0.3)
        if val_metrics['val_spectral_convergence'] > 0:
            status = "✅ PASS" if val_metrics['val_spectral_convergence'] < 0.3 else "❌ FAIL"
            logger.info(f"  Spectral Conv: {val_metrics['val_spectral_convergence']:.4f} (<0.3) {status}")
        
        # 📊 SECONDARY METRICS
        logger.info("📊 SECONDARY METRICS:")
        if val_metrics['val_pesq'] > 0:
            logger.info(f"  PESQ: {val_metrics['val_pesq']:.3f}")
        if val_metrics['val_stoi'] > 0:
            logger.info(f"  STOI: {val_metrics['val_stoi']:.3f}")
        logger.info(f"  SNR Improvement: {val_metrics['val_snr_improvement']:.2f} dB")
        
        logger.info("="*50)
        
        # Save best model
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'args': args,
                # All validation metrics
                'val_loss': val_metrics['val_loss'],
                'val_pesq': val_metrics['val_pesq'],
                'val_stoi': val_metrics['val_stoi'],
                'val_estoi': val_metrics['val_estoi'],
                'val_snr_improvement': val_metrics['val_snr_improvement'],
                'val_lsd': val_metrics['val_lsd'],
                'val_spectral_convergence': val_metrics['val_spectral_convergence'],
                'val_si_sdr': val_metrics['val_si_sdr'],
                'val_si_sdr_improvement': val_metrics['val_si_sdr_improvement'],
            }, args.output_dir / 'best_model.pth')
            logger.info("🎉 Saved best model with improved validation loss!")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, args.output_dir / f'checkpoint_epoch_{epoch + 1}.pth')
    
    logger.info("Training completed!")
    wandb.finish()


if __name__ == '__main__':
    main() 