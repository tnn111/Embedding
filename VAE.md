# VAE.py Development Log

## 2025-11-27 ~14:00: Rewrite for new k-mer format (6-mer, 5-mer based)

### Changes

Rewrote VAE.py to use the new data format from `calculate_kmer_frequencies`:

- **Input**: length(1) + 6-mers(2080) + 5-mers(512) + 4-mers(136) + 3-mers(32) + GC(1) = 2,762 features
- **Output**: 2,761 features (excludes length)

### Architecture (Simple baseline)

- Encoder: 2,762 → 1024 → 512 → latent(256)
- Decoder: 256 → 512 → 1024 → 2,761
- Loss: Simple MSE in log-space + KL divergence
- Latent dimension: 256

### Slice indices

```
Input layout: [length(1), 6-mers(2080), 5-mers(512), 4-mers(136), 3-mers(32), GC(1)]

LENGTH_SLICE = (0, 1)
KMER_6_SLICE = (1, 2081)      # 2,080 features
KMER_5_SLICE = (2081, 2593)   # 512 features
KMER_4_SLICE = (2593, 2729)   # 136 features
KMER_3_SLICE = (2729, 2761)   # 32 features
GC_SLICE = (2761, 2762)       # 1 feature

Output layout (no length): [6-mers(2080), 5-mers(512), 4-mers(136), 3-mers(32), GC(1)]

OUT_KMER_6_SLICE = (0, 2080)
OUT_KMER_5_SLICE = (2080, 2592)
OUT_KMER_4_SLICE = (2592, 2728)
OUT_KMER_3_SLICE = (2728, 2760)
OUT_GC_SLICE = (2760, 2761)
```

### Rationale for 6-mer/5-mer

Previous version used 7-mers (8,192 features) which suffered from:
1. **Sparsity**: Most 7-mers absent in typical 5kb sequences
2. **Information bottleneck**: 8,192 features through 256-512 latent dims
3. **Gated complexity**: Needed separate gate + value prediction

6-mers (2,080 features) and 5-mers (512 features) should have much better coverage - nearly all should appear in typical sequences, making simple log-space MSE viable without gating.

### Parameter count

Rough estimate for simple architecture:
- Encoder: 2,762×1024 + 1024×512 + 512×256×2 ≈ 3.6M
- Decoder: 256×512 + 512×1024 + 1024×2761 ≈ 3.4M
- Total: ~7M parameters

---

## Previous Work Summary (7-mer experiments, archived)

The previous VAE used 7-mers and developed several techniques to handle sparsity:

1. **Gated sparsity**: Separate sigmoid gate (presence prediction) + softplus values
2. **Linear-space output**: 7-mers output in linear space, others in log-space
3. **Final results**: 86.1% gate accuracy, 0.031 4-mer MSE, 0.0035 3-mer MSE

Key learnings:
- Log-space works well for dense k-mers (3-mer, 4-mer)
- Sparse features need special handling
- Gate accuracy plateau around 78-86% due to latent space capacity

These techniques may be revisited if 6-mers show sparsity issues.

---

## 2025-11-27 ~16:30: Hybrid BCE + log-MSE experiments

### Current architecture

- **Loss**: Hybrid BCE (6/5/4-mers) + log-space MSE (3-mers, GC)
- **Scaling**: Both scaled by `OUTPUT_DIM * 100`
- **Parameters**: ~7.1M total (encoder ~3.6M, decoder ~3.5M)
- **Length input**: Log-transformed (advisory only, not reconstructed)

### Results on 0.5M test samples

Training showed instability:
- Recon oscillating: 9,479 to 23,782
- KL: ~500-565
- BCE stable: 6mer=0.0040, 5mer=0.0137, 4mer=0.0420
- MSE unstable: 3mer=0.0286-0.1305, GC=0.0003-0.0026

### Observations

1. BCE components (6/5/4-mers) are stable
2. Log-MSE components (3-mers, GC) cause oscillation
3. Likely gradient scale mismatch between BCE and log-MSE

### Ideas to try with full dataset

1. **Reduce MSE weight**: Log-MSE may need 10-100x smaller weight than BCE
2. **All-BCE or all-MSE**: Remove hybrid approach
3. **Gradient clipping**: Limit per-batch updates
4. **Lower learning rate**: Reduce oscillation

Waiting for full training data before further experiments.

---

## 2025-11-27 ~19:00: Length removal and loss refinements

### Removed length from model

Length was previously an advisory input to the encoder. Removed entirely:
- INPUT_DIM = OUTPUT_DIM = 2761 (no length column)
- Data loading skips column 0 (length) from the 2762-column file
- Simpler architecture, no separate input/output slices needed

Results improved significantly:
- 3-mer MSE: 0.012 → 0.0053
- Training stability: No more oscillation
- KL: ~183 at convergence

### BCE for 3-mers experiment

Tried BCE loss for 3-mers instead of MSE. Result:
- 3-mer BCE: ~0.135 (much higher than other k-mers due to denser distribution)
- KL dropped to ~53 (latent space compressed)
- Overall loss doubled

BCE naturally gives higher values for denser distributions. For 3-mers (~0.03 each), BCE ≈ 0.135. For 6-mers (~0.0005 each), BCE ≈ 0.004. This is mathematically expected, not a bug.

Reverted to MSE for 3-mers since clustering needs higher KL (more spread in latent space).

### Log offset change: eps → 0.5

Changed log-space MSE from `log(x + eps)` to `log(x + 0.5)`:
- Old range: log(eps) to log(1) ≈ -16 to 0
- New range: log(0.5) to log(1.5) ≈ -0.69 to 0.41

Results:
- 3-mer MSE: 0.0053 → 0.0001
- GC MSE: ~0.0000
- KL: ~183 → ~25 (too compressed)

The tighter log range reduces MSE contribution, causing KL to drop.

### β-VAE with β=0.1

To encourage latent space spread for clustering, reduced KL weight from 1.0 to 0.1.
This allows the model to use more of the latent space without being penalized heavily.

Current settings:
- KLWarmupCallback: max_weight = 0.1
- Loss: BCE for 6/5/4-mers, MSE with +0.5 offset for 3-mers and GC

---

## 2025-11-27 ~20:00: Final loss configuration

### Loss functions by feature group

| Feature | Count | Loss | Transform |
|---------|-------|------|-----------|
| 6-mers | 2080 | BCE | clip to [eps, 1-eps] |
| 5-mers | 512 | BCE | clip to [eps, 1-eps] |
| 4-mers | 136 | MSE | log(x + 0.01) |
| 3-mers | 32 | MSE | log(x + 0.01) |
| GC | 1 | MSE | logit: log(x / (1-x)) |

### Loss formula

```python
recon_loss = (bce_6 + bce_5) * OUTPUT_DIM * 100 / 2 + (mse_4 + mse_3 + mse_gc) * OUTPUT_DIM * 100 / 3
```

### Converged values (test dataset, ~500 epochs)

| Metric | Value |
|--------|-------|
| 6-mer BCE | 0.0040 |
| 5-mer BCE | 0.0136 |
| 4-mer MSE | 0.0004 |
| 3-mer MSE | 0.0002 |
| GC MSE | 0.0001 |
| KL | ~493 |
| Recon | ~2480 |

### Reconstruction error analysis

**4-mers (MSE = 0.0004):**
- Typical value: ~0.0074 (1/136)
- RMS log error: ~0.02
- Absolute error: ~0.0003 (~4% relative)

**3-mers (MSE = 0.0002):**
- Typical value: ~0.031 (1/32)
- RMS log error: ~0.014
- Absolute error: ~0.0004 (~1.4% relative)

**GC (MSE = 0.0001 with logit):**
- Typical value: 0.2-0.7
- RMS logit error: ~0.01
- Absolute error: ~0.002-0.003 (<1% relative)

All errors well under 0.01 for dense k-mers and GC.

### Key settings

- β-VAE with β = 0.1 (KL weight)
- Latent dimension: 256
- KL ~493 indicates good latent space utilization for clustering
- Length column removed from input (2761 features total)
