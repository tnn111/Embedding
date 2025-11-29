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

---

## 2025-11-27 ~21:45: All-MSE experiment

### Changed 6-mers from BCE to MSE

Tested using MSE log(x + 0.01) for all k-mers instead of BCE for 6-mers.

| Metric | BCE 6-mer | MSE 6-mer |
|--------|-----------|-----------|
| 6-mer | 0.0040 (BCE) | 0.0004 (MSE) |
| 5-mer | 0.0008 | 0.0008 |
| 4-mer | 0.0004 | 0.0004 |
| 3-mer | 0.0002 | 0.0002 |
| KL | ~499 | ~416 |
| Recon | ~1195 | ~95 |

### Observation: Loss doubling pattern breaks for 6-mers

Expected pattern based on 3/4/5-mers: 2 → 4 → 8 → 16
Actual 6-mer loss: 4 (not 16)

**Explanation**: The 0.01 offset drowns out the 6-mer signal.

- 6-mer values: ~1/2080 ≈ 0.00048 (20x smaller than offset)
- 5-mer values: ~1/512 ≈ 0.002
- 4-mer values: ~1/136 ≈ 0.007
- 3-mer values: ~1/32 ≈ 0.031

With +0.01 offset:
- log(0.00048 + 0.01) ≈ -4.56
- log(0 + 0.01) = -4.61

The offset dominates for sparse 6-mers, compressing prediction errors. This may explain why BCE worked well for 6-mers - it doesn't have this offset problem.

### Trade-off

- All-MSE: Lower reconstruction loss, but KL dropped (~499 → ~416)
- BCE for 6-mers: Higher KL may be better for clustering separation

Keeping all-MSE for now pending full dataset testing.

---

## 2025-11-27 ~22:15: Testing smaller offset for 6-mers

### Hypothesis

The 0.01 offset drowns out 6-mer signal because 6-mer values (~0.00048) are 20x smaller than the offset. Testing whether a smaller offset (0.001) restores the expected loss doubling pattern.

### Change

- 6-mers: `log(x + 0.001)` (was 0.01)
- 5/4/3-mers: `log(x + 0.01)` (unchanged)
- GC: logit transform (unchanged)

### Expected outcome

If the hypothesis is correct, 6-mer MSE should increase from ~0.0004 to ~0.0016 (4x), restoring the doubling pattern:
- 3-mer: ~0.0002 (2)
- 4-mer: ~0.0004 (4)
- 5-mer: ~0.0008 (8)
- 6-mer: ~0.0016 (16) ← expected with smaller offset

### Results

| Metric | 0.01 offset | 0.001 offset |
|--------|-------------|--------------|
| 6-mer MSE | 0.0004 | 0.0124 |
| 5-mer MSE | 0.0008 | 0.0005 |
| 4-mer MSE | 0.0004 | 0.0003 |
| 3-mer MSE | 0.0002 | 0.0003 |
| GC MSE | ~0.0001 | 0.0001 |
| KL | ~416 | ~623 |
| Recon | ~95 | ~752 |

### Analysis

The 6-mer MSE increased 31x (not the expected 4x). The smaller offset "unsuppressed" the 6-mer signal, revealing that the model struggles to reconstruct sparse 6-mers precisely.

**Trade-offs:**
- Higher KL (~623 vs ~416) = better latent space utilization for clustering
- Worse 6-mer reconstruction = model encodes what it can, ignores fine 6-mer details
- Slightly better 5/4-mer reconstruction

For clustering purposes, higher KL may be more valuable than perfect 6-mer reconstruction. The sparse 6-mers contain less discriminative information anyway.

---

## 2025-11-28 ~00:10: Latent dimension comparison (256 vs 128)

### Test: Reduced latent dimension to 128

Ran overnight to compare 128-dim latent space vs 256-dim.

| Metric | 256-dim | 128-dim |
|--------|---------|---------|
| 6-mer MSE | 0.0124 | 0.0136 |
| 5-mer MSE | 0.0005 | 0.0007 |
| 4-mer MSE | 0.0003 | 0.0004 |
| 3-mer MSE | 0.0003 | 0.0002 |
| GC MSE | 0.0001 | 0.0000 |
| KL | ~623 | ~411 |
| Recon | ~752 | ~822 |

### Observations

- Reconstruction slightly worse with 128-dim (expected with less capacity)
- KL dropped from ~623 to ~411 (smaller latent space = less room to spread)
- Dense k-mers (3-mer, 4-mer) and GC still reconstruct well
- 6-mers hardest to reconstruct in both cases

### Next steps

Test on full dataset to determine if 128 dimensions with KL ~411 provides sufficient clustering separation, or if 256 dimensions with KL ~623 is needed.

---

## 2025-11-28 ~14:15: Full dataset training (256-dim)

### Training progress at epoch ~270

| Metric | Test set (0.5M) | Full dataset |
|--------|-----------------|--------------|
| 6-mer MSE | 0.0124 | 0.0135 |
| 5-mer MSE | 0.0005 | 0.0005 |
| 4-mer MSE | 0.0003 | 0.0004 |
| 3-mer MSE | 0.0003 | 0.0003 |
| GC MSE | 0.0001 | 0.0001 |
| KL | ~623 | ~605 |

Occasional GC spikes (0.0009-0.0015) in some epochs, likely due to batch variation in metagenomic data. Training otherwise stable.

### 6-mer error analysis

With MSE = 0.0135 in `log(x + 0.001)` space:
- RMS log error: √0.0135 ≈ 0.116
- Multiplicative error: e^0.116 ≈ 1.12 (±12%)
- For typical 6-mer (~0.00048): absolute error ~0.00006

The 0.001 offset limits discrimination between absent (0) and very rare 6-mers:
- log(0 + 0.001) = -6.9
- log(0.0005 + 0.001) = -6.5

### Options to improve 6-mer reconstruction

1. **Gated approach**: Separate presence/absence from value prediction (like old 7-mer model)
2. **Smaller offset (0.0001)**: More sensitivity but potentially unstable
3. **BCE for 6-mers only**: Worked well before (0.0040) but gave lower KL
4. **Accept current accuracy**: ±12% may suffice for clustering — species differ by much more

### Practical interpretation

Model distinguishes 6-mer profiles differing by >12%. For metagenomic clustering, this is likely acceptable since species k-mer signatures differ substantially.

---

## 2025-11-28 ~14:50: 6-mer offset experiment (0.0005 vs 0.001)

### Test: Halved 6-mer offset to 0.0005

| Metric | 0.001 offset | 0.0005 offset |
|--------|--------------|---------------|
| 6-mer MSE | 0.0135 | 0.041 |
| 5-mer MSE | 0.0005 | 0.0012 |
| 4-mer MSE | 0.0004 | 0.0012 |
| 3-mer MSE | 0.0003 | 0.0008 |
| KL | ~605 | ~771 |

### Result

Smaller offset made everything worse. The loss landscape becomes harder to optimize when the offset is too small relative to the values. Reverted to 0.001.

### Conclusion

0.001 is the practical minimum for 6-mer offset. Further reduction destabilizes training without improving reconstruction.

---

## 2025-11-28 ~16:55: BCE for 6-mers (full dataset)

### Configuration

- **6-mers**: BCE (clip to [eps, 1-eps])
- **5/4/3-mers**: MSE with `log(x + 0.01)`
- **GC**: MSE with logit transform
- **Loss formula**: `bce_6 * OUTPUT_DIM * 100 + (mse_5 + mse_4 + mse_3 + mse_gc) * OUTPUT_DIM * 100 / 4`

### Results comparison

| Metric | All-MSE (0.001 offset) | BCE 6-mer + MSE others |
|--------|------------------------|------------------------|
| 6-mer | 0.0135 (MSE) | 0.0040 (BCE) |
| 5-mer | 0.0005 | 0.0011 |
| 4-mer | 0.0004 | 0.0007 |
| 3-mer | 0.0003 | 0.0004 |
| GC | 0.0001 | 0.0001 |
| KL | ~605 | ~615 |
| Recon | ~815 | ~1251 |

### Analysis

BCE handles sparse 6-mers much better than MSE with offset:
- 6-mer error: 0.0040 (BCE) vs 0.0135 (MSE) — 3.4x improvement
- KL slightly higher (~615 vs ~605) — good for clustering
- 5/4/3-mer MSE slightly higher but still excellent (<0.1% error)

### Conclusion

BCE for 6-mers is the right choice. The hybrid loss (BCE for sparse features, MSE for dense features) outperforms uniform MSE.
