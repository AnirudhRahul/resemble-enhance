# Simple CNN Denoiser Training Script

This script provides a simplified training pipeline for a CNN-based speech denoiser using the existing `train_dataset` structure. It includes comprehensive validation metrics and Weights & Biases logging, optimized for 48kHz high-quality audio.

## Features

- **Custom CNN Architecture**: Simple encoder-decoder CNN optimized for 48kHz audio
- **Custom Dataloaders**: Efficient loading of clean speech and noise from `train_dataset`
- **Comprehensive Metrics**: 8 validation metrics including PESQ, STOI, ESTOI, SI-SDR, LSD, and more
- **48kHz Native Support**: Full pipeline designed for high-quality 48kHz audio processing
- **W&B Integration**: Complete experiment tracking with proper metric graphing
- **Robust Training**: Gradient clipping, learning rate scheduling, and comprehensive checkpointing

## Installation

1. Install additional dependencies:
```bash
pip install -r simple_denoiser_requirements.txt
```

2. Set up Weights & Biases (optional but recommended):
```bash
wandb login
```

## Dataset Structure

The script expects the following directory structure (as created by the main project):
```
train_dataset/
├── fg/en/          # Clean speech files
│   └── *.wav
└── bg/en/          # Noise files
    └── *.wav
```

## Usage

### Basic Training
```bash
python simple_denoiser_train.py
```

### Advanced Training with Custom Parameters
```bash
python simple_denoiser_train.py \
    --clean_dir train_dataset/fg/en \
    --noise_dir train_dataset/bg/en \
    --output_dir runs/my_denoiser \
    --batch_size 32 \
    --epochs 50 \
    --lr 1e-3 \
    --sample_rate 48000 \
    --segment_length 2.0 \
    --loss_type multi_stft \
    --wandb_project my-denoiser-project
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--clean_dir` | `train_dataset/fg/en` | Directory containing clean speech files |
| `--noise_dir` | `train_dataset/bg/en` | Directory containing noise files |
| `--output_dir` | `runs/simple_denoiser` | Output directory for checkpoints |
| `--batch_size` | `16` | Training batch size |
| `--epochs` | `100` | Number of training epochs |
| `--lr` | `1e-3` | Learning rate |
| `--sample_rate` | `48000` | Audio sample rate (Hz) |
| `--segment_length` | `2.0` | Audio segment length (seconds) |
| `--loss_type` | `multi_stft` | Loss function (multi_stft/mse) |
| `--wandb_project` | `simple-denoiser` | W&B project name |
| `--device` | `auto` | Device (cuda/cpu) |

## Model Architecture

The `SimpleCNNDenoiser` uses an encoder-decoder architecture:

### Encoder
- 4 convolutional layers with increasing channels (64 → 128 → 256 → 512)
- Batch normalization and ReLU activation
- Max pooling for downsampling

### Decoder
- 2 transposed convolutional layers for upsampling
- 2 regular convolutional layers for refinement
- Sigmoid activation for mask prediction

### Output
- Predicted mask applied to noisy spectrogram
- Mask values between 0 and 1

## Validation Metrics

The script tracks several important metrics for speech enhancement evaluation:

## 🎯 PRIMARY VALIDATION METRICS

These are the key metrics to focus on for speech enhancement evaluation:

### 1. ESTOI (Extended STOI) - Speech Intelligibility
- **Range**: 0 to 1 (higher is better)
- **🎯 Target**: **≥ 0.85**
- **Purpose**: Works natively at 48kHz; better than STOI for wide-band speech
- **Priority**: **PRIMARY** - Most important for speech quality

### 2. SI-SDR (Scale-Invariant Signal-to-Distortion Ratio) - Noise Removal
- **Range**: -∞ to +∞ dB (higher is better)
- **🎯 Target**: **> 10 dB**
- **Purpose**: Modern replacement for SNR; measures noise-removal strength & artifacts
- **Priority**: **PRIMARY** - Key metric for denoising effectiveness

### 3. Log-Spectral Distance (LSD) - Spectral Fidelity
- **Range**: 0 to ∞ (lower is better)
- **🎯 Target**: **< 1 dB**
- **Purpose**: Captures high-freq errors the ear notices most; measures "crispness"
- **Priority**: **PRIMARY** - Critical for audio quality

### 4. Spectral Convergence - Spectral Accuracy
- **Range**: 0 to ∞ (lower is better)
- **🎯 Target**: **< 0.3**
- **Purpose**: Complement to LSD; measures spectral reconstruction accuracy
- **Priority**: **PRIMARY** - Works well with 48kHz spectrograms

## 📊 SECONDARY METRICS

### 5. PESQ (Perceptual Evaluation of Speech Quality)
- **Range**: -0.5 to 4.5 (higher is better)
- **Good values**: > 2.5
- **Note**: Automatically downsamples 48kHz to 16kHz for calculation

### 6. STOI (Short-Time Objective Intelligibility)
- **Range**: 0 to 1 (higher is better)
- **Good values**: > 0.8
- **Purpose**: Standard intelligibility metric (ESTOI is preferred)

### 7. SNR Improvement
- **Range**: -∞ to +∞ dB (higher is better)
- **Good values**: > 5 dB improvement
- **Purpose**: Traditional noise reduction measure (SI-SDR is preferred)

### 8. Training Loss (Multi-scale STFT or MSE)
- **Purpose**: Low-level loss for training
- **Note**: Use in the loss, but don't rely on it alone for evaluation

## Training Process

### Data Augmentation
- **Random SNR**: -5 to 20 dB during training
- **Random cropping**: 2-second segments from longer audio
- **Dynamic mixing**: Clean speech + random noise per batch

### Optimization
- **Optimizer**: Adam with learning rate 1e-3
- **Scheduler**: OneCycleLR for learning rate cycling
- **Gradient clipping**: Max norm of 1.0
- **Loss function**: Multi-scale STFT (recommended) or MSE
  - **Multi-scale STFT**: Uses multiple STFT scales for better spectral reconstruction
  - **MSE**: Simple MSE between predicted and clean spectrograms

### Checkpointing
- **Best model**: Saved based on lowest validation loss
- **Periodic checkpoints**: Every 10 epochs
- **Metrics saved**: PESQ, STOI, SNR improvement

## Monitoring with W&B

The script logs comprehensive metrics to Weights & Biases:

### Training Metrics
- Training loss (per step and per epoch)
- Learning rate scheduling
- Gradient norms

### Validation Metrics
- Validation loss
- PESQ scores
- STOI scores
- SNR improvement
- Before/after SNR values

### Model Information
- Total parameters
- Trainable parameters
- Model architecture details

### Visualization
- Loss curves
- Metric trends over epochs
- Hyperparameter importance
- Real-time primary metric pass/fail status

### Training Console Output
The training script provides clear feedback on primary metric performance:

```
=== Epoch 25/100 Results ===
Train Loss: 0.1234
Val Loss: 0.0987

🎯 PRIMARY METRICS:
  ESTOI: 0.872 (≥0.85) ✅ PASS
  SI-SDR: 12.3 dB (>10) ✅ PASS  
  LSD: 0.84 dB (<1.0) ✅ PASS
  Spectral Conv: 0.275 (<0.3) ✅ PASS

📊 SECONDARY METRICS:
  PESQ: 2.45
  STOI: 0.823
  SNR Improvement: 8.7 dB
```

## Performance Optimization Tips

### Hardware
- **GPU**: Use CUDA if available (automatic detection)
- **Memory**: 16GB+ RAM recommended for large datasets
- **Storage**: SSD recommended for faster data loading

### Training
- **Batch size**: Increase if GPU memory allows (try 32, 64)
- **Num workers**: Set to number of CPU cores for data loading
- **Mixed precision**: Add `torch.cuda.amp` for faster training

### Data
- **Sample rate**: 48kHz for high-quality audio (matches your dataset)
- **Segment length**: 2-3 seconds balances context and memory usage
- **Validation split**: Use separate validation set for better evaluation
- **STFT parameters**: n_fft=2048, hop_length=512 (optimized for 48kHz)

## Expected Results

### Target Performance Goals

#### 🎯 PRIMARY METRICS (Key Targets)
- **ESTOI**: **≥ 0.85** (speech intelligibility)
- **SI-SDR**: **> 10 dB** (noise removal strength)
- **LSD**: **< 1 dB** (spectral fidelity/"crispness")
- **Spectral Convergence**: **< 0.3** (spectral accuracy)

#### 📊 Secondary Metrics (Expected Range)
- **PESQ**: 2.0 - 3.5 (perceptual quality)
- **STOI**: 0.7 - 0.9 (standard intelligibility)
- **SNR Improvement**: 5 - 15 dB (traditional noise measure)

#### 🚀 Excellent Performance Indicators
- **ESTOI > 0.90** + **SI-SDR > 15 dB** + **LSD < 0.7** = Outstanding model
- **All 4 primary metrics hit targets** = Production-ready quality

### Training Time
- **GPU (RTX 3080)**: ~2-4 hours for 50 epochs (depends on dataset size)
- **CPU**: 10-20x slower than GPU

## Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce batch size
   - Decrease segment length
   - Use gradient checkpointing

2. **Slow Training**
   - Increase num_workers in DataLoader
   - Use GPU if available
   - Reduce validation frequency

3. **Poor Metrics**
   - Check data quality and alignment
   - Verify SNR range is reasonable
   - Increase training epochs
   - Try different learning rates (1e-4, 5e-4)

4. **PESQ Errors**
   - PESQ automatically handles 48kHz→16kHz downsampling
   - Check audio length (minimum 0.25 seconds required)
   - Verify clean/noisy audio alignment

5. **48kHz Specific Issues**
   - Ensure sufficient GPU memory (48kHz needs ~3x more than 16kHz)
   - Check librosa version for proper 48kHz resampling
   - SI-SDR may be sensitive to audio length alignment

### Debug Mode
Add `--wandb_project debug` to run without W&B logging.

## Customization

### Model Architecture
Modify `SimpleCNNDenoiser` class to experiment with:
- Different kernel sizes
- More/fewer layers
- Skip connections
- Attention mechanisms

### Loss Functions
Try different loss functions in `train_epoch()`:
- L1 Loss: `nn.L1Loss()`
- Spectral Loss: Custom STFT-based loss
- Perceptual Loss: Pre-trained network features

### Data Augmentation
Enhance `AudioDataset` with:
- Pitch shifting
- Time stretching
- Reverb simulation
- Multiple noise sources

## Citation

If you use this training script in your research, please cite the original Resemble Enhance paper and repository.

## License

This script follows the same license as the main Resemble Enhance project. 