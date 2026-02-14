# embedding Development Log

## Purpose

Standalone script to generate VAE latent embeddings from k-mer frequency data.
Acts as a pure filter: reads NumPy array from stdin, writes embeddings to stdout.

---

## 2025-12-03 ~18:30: Updated for new data format

### Changes

Updated script to match new `calculate_kmer_frequencies` output and VAE encoder:

**Old format:**
- Input: 2,762 columns (length + 6-mers + 5-mers + 4-mers + 3-mers + GC)
- Encoder: `vae_encoder_final.keras` (256-dim latent)
- No CLR transformation (was applied elsewhere)

**New format:**
- Input: 2,773 columns (length + 6-mers + 5-mers + 4-mers + 3-mers + 2-mers + 1-mers)
- Encoder: `vae_encoder_best.keras` (384-dim latent, beta=0.05)
- CLR transformation applied before encoding

### Code changes

1. Updated docstring to reflect new column counts
2. Changed encoder path from `vae_encoder_final.keras` to `vae_encoder_best.keras`
3. Updated column slice from `[:, 1:]` to `[:, 1:2773]`
4. Added `clr_transform()` function
5. Output is now 385 columns: length (col 0) + 384-dimensional embeddings (cols 1-384)

### Output format

| Column | Content |
|--------|---------|
| 0 | Sequence length |
| 1-384 | VAE latent embeddings (z_mean) |

### Usage

```bash
cat kmers.npy | ./embedding > embeddings.npy
```

Input must be in the format output by `calculate_kmer_frequencies` (2,773 columns).

In a notebook:
```python
data = np.load('embeddings.npy')
lengths = data[:, 0]
embeddings = data[:, 1:]
```
