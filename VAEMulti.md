# VAEMulti Development Log

## 2025-12-02 ~15:00: Initial Implementation

### Request
Build a VAE using 6-mers through 1-mers (not 7-mers) from Data/all_kmers.npy.
- Dense layers with batch normalization
- Symmetric encoder/decoder
- CLR transformation with MSE loss

### Data Layout in all_kmers.npy (10,965 columns)
- Column 0: row index
- Columns 1-8192: 7-mers (8,192 features) - NOT USED
- Columns 8193-10272: 6-mers (2,080 features)
- Columns 10273-10784: 5-mers (512 features)
- Columns 10785-10920: 4-mers (136 features)
- Columns 10921-10952: 3-mers (32 features)
- Columns 10953-10962: 2-mers (10 features)
- Columns 10963-10964: 1-mers (2 features)

**Total input features: 2,772** (columns 8193-10964)

### Architecture

**Encoder:**
```
Input (2772,)
    -> Dense(1024) -> BatchNorm -> LeakyReLU(0.2)
    -> Dense(512) -> BatchNorm -> LeakyReLU(0.2)
    -> z_mean(256), z_log_var(256)
```

**Decoder:**
```
Input (256,)
    -> Dense(512) -> BatchNorm -> LeakyReLU(0.2)
    -> Dense(1024) -> BatchNorm -> LeakyReLU(0.2)
    -> Dense(2772)
```

### Parameters (from model summary)
- Encoder: 3,633,152 (13.86 MB)
- Decoder: 3,504,340 (13.37 MB)
- Total: ~7.1M parameters

### Test Run (50k samples, 3 epochs)
```
Epoch 1: Recon: 32182.76, KL: 1556.31, MSE: 11.61
Epoch 2: Recon: 9258.21, KL: 867.71, MSE: 3.34
Epoch 3: Recon: 3896.94, KL: 475.37, MSE: 1.41
```
KL is staying meaningful and MSE is dropping well.

### Key Differences from VAE.py (7-mer version)
1. Input is flat (2772,) not (8192, 1) - no reshape needed
2. Smaller input dimension means smaller first layer
3. Uses columns 8193-10964 instead of 1-8192
4. Model files prefixed with `vae_multi_` instead of `vae_`

### Preprocessing
- CLR (Centered Log-Ratio) transformation applied in-place
- Pseudocount of 1e-6 to avoid log(0)

---

## 2025-12-02 ~15:55: Added per-k-mer MSE breakdown

### Change
Added breakdown of MSE by k-mer size to monitor reconstruction quality at each scale.

### Output format
```
MSE: 1.41 [6mer=1.52, 5mer=1.21, 4mer=0.98, 3mer=0.76, 2mer=0.45, 1mer=0.12]
```

### K-mer feature ranges (local indices in 2772-dim input)
- 6mer: 0-2080 (2080 features)
- 5mer: 2080-2592 (512 features)
- 4mer: 2592-2728 (136 features)
- 3mer: 2728-2760 (32 features)
- 2mer: 2760-2770 (10 features)
- 1mer: 2770-2772 (2 features)

---

## 2025-12-02 ~16:15: Training observations and analysis

### Results at epoch ~100
```
Recon: 3366.99, KL: 681.81 (w=0.1000), Val: 3397.95, MSE: 1.21 [6mer=1.56, 5mer=0.22, 4mer=0.06, 3mer=0.02, 2mer=0.01, 1mer=0.01]
```

### Key observation: 6-mers dominate reconstruction error
| K-mer | MSE  | % of total |
|-------|------|------------|
| 6mer  | 1.56 | ~95%       |
| 5mer  | 0.22 | ~4%        |
| 4mer  | 0.06 | <1%        |
| 3-1mer| 0.01-0.02 | tiny |

**Interpretation:**
- Shorter k-mers (1-4) are nearly perfectly reconstructed
- 6-mers are the bottleneck - 2,080 features compressed to 256 latent dims is lossy
- Model has converged; improvements are marginal

### Options to improve reconstruction while maintaining structured latent space

1. **Increase latent dimension** (256 → 384 or 512)
   - Simplest change, directly addresses compression bottleneck
   - Recommended first step

2. **Add encoder/decoder capacity**
   - Wider layers (1024→2048) or add third layer
   - More parameters, slower training

3. **Learned prior** (VampPrior, mixture of Gaussians)
   - More expressive than N(0,1) prior

4. **Hierarchical VAE**
   - Separate latent spaces for different k-mer scales

5. **Cyclical annealing**
   - Repeatedly ramp beta up and down during training

### Effect of lowering beta (KL weight)
- Lower beta (e.g., 0.075 instead of 0.1) → better reconstruction but less structured latent space
- Higher beta → more regularized latent space but worse reconstruction
- At beta=0, just a regular autoencoder (no KL term)
