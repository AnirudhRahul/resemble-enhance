#!/usr/bin/env python3
"""Debug script to check audio shapes and memory requirements."""

import torch
import numpy as np

def calculate_memory_usage(batch_size=8, segment_length=8.0, sample_rate=48000, n_fft=2048, hop_length=512):
    """Calculate approximate memory usage for training."""
    
    # Audio samples
    audio_samples = int(segment_length * sample_rate)
    
    # Spectrogram dimensions
    n_freq = n_fft // 2 + 1
    n_frames = (audio_samples - n_fft) // hop_length + 1
    
    print(f"=== Configuration ===")
    print(f"Batch size: {batch_size}")
    print(f"Segment length: {segment_length} seconds")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Audio samples: {audio_samples}")
    print(f"STFT params: n_fft={n_fft}, hop_length={hop_length}")
    
    print(f"\n=== Spectrogram Dimensions ===")
    print(f"Frequency bins: {n_freq}")
    print(f"Time frames: {n_frames}")
    print(f"Spectrogram shape per sample: ({n_freq}, {n_frames})")
    
    # Memory calculations (float32 = 4 bytes)
    bytes_per_float = 4
    
    # Per batch memory
    audio_memory = batch_size * audio_samples * bytes_per_float * 4  # clean + noisy audio * 2 (cpu + gpu)
    spec_memory = batch_size * 1 * n_freq * n_frames * bytes_per_float * 4  # clean + noisy spec * 2
    
    # Model memory (rough estimate)
    model_params = 4_248_833  # From your output
    model_memory = model_params * bytes_per_float * 3  # params + gradients + optimizer states
    
    # Multi-scale STFT loss memory
    stft_memory = 0
    for fft_size in [512, 1024, 2048]:
        stft_freq = fft_size // 2 + 1
        stft_frames = (audio_samples - fft_size) // (fft_size // 4) + 1
        stft_memory += batch_size * stft_freq * stft_frames * bytes_per_float * 4
    
    total_memory = audio_memory + spec_memory + model_memory + stft_memory
    
    print(f"\n=== Memory Usage Estimates ===")
    print(f"Audio data: {audio_memory / 1024**3:.2f} GB")
    print(f"Spectrogram data: {spec_memory / 1024**3:.2f} GB")
    print(f"Model + optimizer: {model_memory / 1024**3:.2f} GB")
    print(f"Multi-scale STFT loss: {stft_memory / 1024**3:.2f} GB")
    print(f"Total estimated: {total_memory / 1024**3:.2f} GB")
    
    print(f"\n=== Recommendations ===")
    if segment_length > 4.0:
        print("⚠️  WARNING: Segment length > 4 seconds may cause memory issues!")
        print("   Consider reducing to 2-3 seconds for stable training.")
    
    if total_memory > 8 * 1024**3:
        print("⚠️  WARNING: Estimated memory usage exceeds 8GB!")
        print("   Reduce batch size or segment length.")
        
        # Suggest alternative configs
        alt_segment = 2.0
        alt_samples = int(alt_segment * sample_rate)
        alt_frames = (alt_samples - n_fft) // hop_length + 1
        alt_memory = (batch_size * alt_samples * 8 + 
                     batch_size * n_freq * alt_frames * 8 + 
                     model_memory + stft_memory // 4) / 1024**3
        
        print(f"\n   Suggested alternative:")
        print(f"   --segment_length {alt_segment} (reduces memory to ~{alt_memory:.1f} GB)")
        
        if batch_size > 16:
            alt_batch = 16
            alt_memory2 = total_memory * (alt_batch / batch_size) / 1024**3
            print(f"   --batch_size {alt_batch} (reduces memory to ~{alt_memory2:.1f} GB)")

if __name__ == "__main__":
    # Your current configuration
    calculate_memory_usage(
        batch_size=8,
        segment_length=8.0,
        sample_rate=48000,
        n_fft=2048,
        hop_length=512
    )
    
    print("\n" + "="*50)
    print("\n=== Optimal Configuration for 48kHz ===")
    calculate_memory_usage(
        batch_size=16,
        segment_length=2.0,
        sample_rate=48000,
        n_fft=2048,
        hop_length=512
    ) 