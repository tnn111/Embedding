# VAE.py Development Log

## 2025-11-27: Rewrite for new k-mer format (6-mer, 5-mer based)

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
