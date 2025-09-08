#!/usr/bin/env python3
"""Test script to verify small dataset sampling works correctly."""

from pathlib import Path
from simple_denoiser_train import AudioDataset

# Test dataset creation with small sample size
print("Testing dataset with max_samples=10...")

dataset = AudioDataset(
    clean_dir=Path("train_dataset/fg/en"),
    noise_dir=Path("train_dataset/bg/en"),
    max_samples=10,
    training=True
)

print(f"\nDataset info:")
print(f"- Total clean files available: {len(dataset.all_clean_files)}")
print(f"- Sampled for this epoch: {len(dataset.clean_files)}")
print(f"- Dataset length: {len(dataset)}")

print("\nTesting resampling...")
original_files = dataset.clean_files.copy()
dataset.resample_epoch_data()
new_files = dataset.clean_files

# Check if files changed
files_changed = original_files != new_files
print(f"Files changed after resampling: {files_changed}")

print("\nTesting data loading...")
sample = dataset[0]
print(f"Sample keys: {sample.keys()}")
print(f"fg_wav shape: {sample['fg_wav'].shape}")
print(f"bg_wav shape: {sample['bg_wav'].shape}")

print("\nTest completed successfully!") 