# Trained Model Checkpoints

This repository contains pre-trained model checkpoints for CaloVAR and CaloVQ-VAE calorimeter simulation models.

## Repository Structure

```
├── calo_var/           # CaloVAR transformer model checkpoints
│   ├── OS/             # Original Scale configuration
│   ├── RL/             # R-token with Label conditioning
│   └── RS/             # R-token with Scale0 conditioning
│
└── calo_vq_vae/        # CaloVQ-VAE encoder-decoder checkpoints
    ├── new_arch_norm/
    ├── new_arch_norm_small_scale/
    └── old_arch_not_norm_small_scale/
```

## Downloading the Model Files

This repository uses **Git Large File Storage (LFS)** to store the model checkpoint files (`.pth` and `.ckpt`).

### Prerequisites

1. Install Git LFS: https://git-lfs.github.com/

   - **Windows**: Download the installer from the website or use `winget install GitHub.GitLFS`
   - **macOS**: `brew install git-lfs`
   - **Linux**: `sudo apt install git-lfs` (Debian/Ubuntu) or equivalent

2. Initialize Git LFS (one-time setup):
   ```bash
   git lfs install
   ```

### Cloning the Repository

When you clone this repository, Git LFS will automatically download the large files:

```bash
git clone https://github.com/wojfran/trained_models_thesis_fw.git
```

### If Files Appear as Pointers

If the model files appear as small text files (LFS pointers) instead of the actual model weights, run:

```bash
git lfs pull
```

This will download all LFS-tracked files.

### Downloading Specific Files

To download only specific LFS files without pulling everything:

```bash
git lfs pull --include="calo_var/OS/*"
```

## File Formats

- `.pth` - PyTorch model state dictionaries (CaloVAR)
- `.ckpt` - PyTorch Lightning checkpoints (CaloVQ-VAE)

## License

See [LICENSE](LICENSE) for details.
