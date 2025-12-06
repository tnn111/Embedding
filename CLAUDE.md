# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project implements a Variational Autoencoder (VAE) for embedding metagenomic DNA sequences based on k-mer frequency distributions. The VAE compresses 2,761-dimensional k-mer frequency vectors into 256-dimensional latent embeddings suitable for clustering and similarity search.

## Running Commands

```bash
# Train VAE on k-mer frequency data
uv run VAE.py -i Data/all_multimer_frequencies_l5000_shuffled.npy -e 100

# Load embeddings into ChromaDB (uses PEP 723 inline dependencies)
./create_and_load_db Data/all_multimer_frequencies_l5000_shuffled.txt

# Run Jupyter notebooks for analysis
uv run jupyter notebook
```

## Data Format

Input k-mer frequency files have 2,762 columns:
- Column 0: sequence length (skipped by VAE)
- Columns 1-2080: 6-mer frequencies (2,080 features)
- Columns 2081-2592: 5-mer frequencies (512 features)
- Columns 2593-2728: 4-mer frequencies (136 features)
- Columns 2729-2760: 3-mer frequencies (32 features)
- Column 2761: GC content (1 feature)

The VAE processes columns 1-2761 (2,761 features total).

## Architecture

### VAE (VAE.py)

- **Encoder**: 2761 → 1024 → 512 → 256 (latent)
- **Decoder**: 256 → 512 → 1024 → 2761
- **Latent dimension**: 256
- **KL weight (β)**: 0.1 (β-VAE for better clustering)

### Loss Function (Hybrid)

| Feature Group | Loss Type | Transform |
|---------------|-----------|-----------|
| 6-mers | BCE | clip to [eps, 1-eps] |
| 5/4/3-mers | MSE | log(x + 0.01) |
| GC | MSE | logit: log(x/(1-x)) |

BCE handles sparse 6-mers better; MSE in log-space works for denser k-mers.

### Key Constants (VAE.py)

```python
INPUT_DIM = OUTPUT_DIM = 2761
LATENT_DIM = 256
KMER_6_SLICE = (0, 2080)
KMER_5_SLICE = (2080, 2592)
KMER_4_SLICE = (2592, 2728)
KMER_3_SLICE = (2728, 2760)
GC_SLICE = (2760, 2761)
```

### Custom Keras Layers

When loading saved models, register these custom objects:
- `Sampling`: Reparameterization trick for VAE
- `ClipLayer`: Clips z_log_var to [-20, 2]
- `SliceLayer`: Tensor slicing along last axis

## ChromaDB Integration

The `create_and_load_db` script loads VAE embeddings into ChromaDB:
- Collection: `shrub_of_life`
- Distance metric: cosine similarity
- Metadata: sequence length
- ~4.8M sequences in full dataset

## Model Checkpoints

- `vae_best.keras` / `vae_final.keras`: Full VAE model
- `vae_encoder_best.keras` / `vae_encoder_final.keras`: Encoder only (for inference)
- `vae_decoder_best.keras` / `vae_decoder_final.keras`: Decoder only
- `vae_history.pkl`: Training history

## Mandatory Logging (CRITICAL)

Whenever making changes to a Python file (X.py), you MUST update the corresponding notes file (X.md) in the SAME response or immediately after. Do NOT wait to be asked. This includes:
- Architecture changes
- Bug fixes
- New features
- Design decisions
- Training observations
- Parameter changes
- Discussions about trade-offs or future directions

Never skip this step. If you forget, add the notes as soon as you realize.

## Development Notes

- Uses JAX backend for Keras (`KERAS_BACKEND = 'jax'`)
- Training resumes automatically if `vae_best.keras` exists
- KL warmup over 5 epochs, skipped when resuming
- Notebooks in `clustering.ipynb`, `umap.ipynb`, `shrub_of_life.ipynb` for analysis
