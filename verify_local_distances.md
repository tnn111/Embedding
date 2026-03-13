# verify_local_distances.py — Development Notes

## Purpose
Evaluates embedding quality by measuring Spearman rank correlation between latent-space Euclidean distances and k-mer MSE for nearest neighbors.

## Key Parameters
- `--sample-size 50000` — CRITICAL: standardized pool size. Spearman is pool-size dependent (larger pools → higher Spearman). Default is 10000 but 50000 is used for all cross-model comparisons.
- `--bootstrap 10000` — Added 2026-03-12. Resamples over queries (not pairs) because 50 neighbors per query are correlated.
- `--metric euclidean` — Default. Euclidean outperforms cosine by +0.076.

## Pool Size Sensitivity (discovered 2026-03-12)

Spearman correlation increases monotonically with pool size because larger pools provide more neighbor candidates at diverse distances:

| Pool size | SFE_SE_5 self-eval | NCBI_5 on SFE_SE_5 |
|-----------|-------------------|---------------------|
| 10,000 | 0.491 | — |
| 50,000 | 0.691 | 0.831 |
| 200,000 | 0.781 | — |
| 477,677 (full val) | 0.822 | 0.881 |

Different models have different sensitivity to pool size. The effect is larger for models with more compressed latent spaces (SFE_SE_5) than for well-structured ones (NCBI_5, where 50K captures 76% of the 65.6K validation set).

## Changes

### 2026-03-12: Added bootstrap CIs
- `--bootstrap` flag (default 10000 resamples, 0 to skip)
- Resamples over queries (natural resampling unit), not individual pairs
- Query-level resampling accounts for within-query correlation (50 neighbors share the same query)
- Output: 95% CI using percentile method

### 2026-02-12: Added --metric flag
- Supports `euclidean` (default) and `cosine`

### 2026-02-05: CLR fix and z_mean inference
- Per-group CLR with Jeffreys prior (0.5/n_features per group)
- Fixed to use z_mean (deterministic) instead of sampled z
