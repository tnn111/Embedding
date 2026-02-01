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

---

## 2025-12-02 ~16:25: Increased latent dimension to 384

Changed LATENT_DIM from 256 to 384 to improve 6-mer reconstruction while maintaining structured latent space.

---

## 2025-12-02 ~16:30: Decision to focus on 6-mers and below (not 7-mers)

### Use case
- Separate organisms to ~species level
- Separate plasmids and viruses from chromosomal DNA

### Why 6-mers through 1-mers are sufficient

**For species-level taxonomy:**
- Studies show 4-6 mers capture most taxonomic signal
- Species have distinct GC content, codon usage, oligonucleotide frequencies

**For plasmids:**
- Different compositional signatures than host chromosomes
- 4-6 mers commonly used in plasmid detection tools

**For viruses:**
- Distinctive k-mer profiles due to codon adaptation, compact genomes
- Often extreme GC content
- Phages cluster near bacterial hosts

### Why NOT add 7-mers
1. 6-mers already dominate reconstruction error (95% of MSE)
2. Information redundancy - 7-mers contain constituent 6-mers
3. Would nearly 4x input dimension (2,772 → 10,964)
4. 7-mers mainly help with strain-level differentiation (not needed)

### Expected latent space structure
- Bacteria clustering by phylum/class, species as sub-clusters
- Plasmids clustering separately or near typical hosts
- Viruses forming distinct clusters

---

## 2025-12-02 ~16:40: Systematic parameter exploration

### Run 1: Latent=384, Beta=0.1 (500 epochs)

**Final results:**
```
Val loss: 3001.15, KL: 1059, MSE: 1.054
[6mer=1.351, 5mer=0.203, 4mer=0.054, 3mer=0.015, 2mer=0.009, 1mer=0.006]
```

**Comparison to 256 latent (at epoch ~100):**
- 6mer MSE: 1.56 → 1.35 (13% improvement)
- KL: 680 → 1060 (more informative latent space)
- Larger latent dimension helped capture more 6-mer information

### Run 2: Latent=384, Beta=0.2 (500 epochs)

**Final results:**
```
Val loss: 3077.12, KL: 814, MSE: 1.063
[6mer=1.361, 5mer=0.207, 4mer=0.057, 3mer=0.015, 2mer=0.009, 1mer=0.006]
```

**Comparison to Beta=0.1:**
| Metric | Beta=0.1 | Beta=0.2 |
|--------|----------|----------|
| Val loss | 3001 | 3077 |
| KL | 1059 | 814 |
| MSE | 1.054 | 1.063 |
| 6mer | 1.351 | 1.361 |

- Higher beta → lower KL (more regularized latent space)
- Slightly worse reconstruction (as expected)
- Differences are small - both runs converged well

### Run 3: Latent=384, Beta=0.05 (500 epochs)

**Final results:**
```
Val loss: 2899.36, KL: 1261, MSE: 1.030
[6mer=1.319, 5mer=0.203, 4mer=0.053, 3mer=0.015, 2mer=0.008, 1mer=0.005]
```

**Comparison across all beta values:**
| Metric | Beta=0.05 | Beta=0.1 | Beta=0.2 |
|--------|-----------|----------|----------|
| Val loss | 2899 | 3001 | 3077 |
| KL | 1261 | 1059 | 814 |
| MSE | 1.030 | 1.054 | 1.063 |
| 6mer | 1.319 | 1.351 | 1.361 |

**Key findings:**
- Beta=0.05 gives best reconstruction (lowest MSE, lowest 6mer error)
- KL is healthy at 1261 (not collapsed, very expressive latent space)
- Lower beta → more informative latent space + better reconstruction
- For vector DB retrieval, beta=0.05 looks optimal

**Use case clarification:**
- Goal: Insert into vector DB, find N closest sequences to a query
- Not clustering per se - nearest-neighbor retrieval
- Need meaningful local distances, not necessarily global structure

### Run 4: Latent=384, Beta=0.03 (500 epochs - not fully converged)

**Final results:**
```
Val loss: 2914.22 (best: 2910.42), KL: 1613, MSE: 1.042
[6mer=1.333, 5mer=0.208, 4mer=0.061, 3mer=0.017, 2mer=0.010, 1mer=0.009]
```

**Comparison:**
| Metric | Beta=0.03 | Beta=0.05 | Beta=0.1 | Beta=0.2 |
|--------|-----------|-----------|----------|----------|
| Val loss | 2910* | 2899 | 3001 | 3077 |
| KL | 1613 | 1261 | 1059 | 814 |
| MSE | 1.040* | 1.030 | 1.054 | 1.063 |
| 6mer | 1.331 | 1.319 | 1.351 | 1.361 |

*Still improving at epoch 500, not fully converged

**Observations:**
- More variance in metrics compared to beta=0.05 (still improving at epoch 498)
- KL jumped to ~1600 (very expressive, approaching autoencoder territory)
- MSE slightly worse than beta=0.05 despite lower regularization
- May need more epochs to converge, or loss landscape is noisier with low beta

**Interpretation:**
Beta=0.05 appears to be near optimal - lower beta (0.03) doesn't improve reconstruction and makes training less stable.

---

## 2025-12-03 ~12:45: Local distance analysis

### Concern
Does the latent space preserve local structure? For vector DB retrieval, we need:
- Close in latent space → similar k-mer profiles
- Distance ranking to be meaningful

### How beta affects local distances

**With beta=0.05 (KL~1260):**
- Encoder uses latent dimensions expressively
- Similar k-mer profiles → similar latent vectors
- Distances reflect reconstruction similarity

**Potential concerns with low beta:**
1. Discontinuities/holes in latent space (less regularization)
2. Non-uniform density (okay for retrieval)
3. Distance scale variation across regions

**Why probably okay for retrieval:**
- Finding N closest neighbors (not using distance threshold)
- Ranking by distance should work
- KL=1260 is still substantial regularization

### Verification script: verify_local_distances.py
Tests whether "close in latent space" = "similar k-mer profiles" by:
1. Taking random sequences from validation set
2. Finding K nearest neighbors in latent space
3. Computing actual k-mer MSE between query and neighbors
4. Checking correlation between latent distance and k-mer similarity

### Results (beta=0.03 model, 10k samples)
```
Pearson correlation:  r = 0.9231 (p = 0.00e+00)
Spearman correlation: r = 0.9579 (p = 0.00e+00)

K-mer MSE by neighbor rank (latent space):
  Top  1 neighbors: MSE = 2.03 ± 1.26
  Top  5 neighbors: MSE = 2.09 ± 1.28
  Top 10 neighbors: MSE = 2.12 ± 1.28
  Top 20 neighbors: MSE = 2.16 ± 1.29
  Top 50 neighbors: MSE = 2.24 ± 1.33

Random baseline: MSE = 6.19 ± 3.36
```

**Random pairs MSE distribution:**
| Stat | Value |
|------|-------|
| Min | 0.17 |
| Max | 25.91 |
| Mean | 6.16 |
| Median | 5.50 |
| Std | 3.31 |

**Interpretation:**
- **STRONG correlation (r=0.96)** - latent distances reliably predict k-mer similarity
- Nearest neighbors (MSE ~2.0) are much better than median random pair (5.5)
- Random pairs range from 0.17 (lucky match) to 25.91 (very dissimilar - likely different domains of life)
- MSE increases gradually with rank (as expected)
- Local structure is well-preserved for retrieval

### Comparison: Local distance quality across beta values

| Metric | Beta=0.03 | Beta=0.05 | Beta=0.1 |
|--------|-----------|-----------|----------|
| Spearman r | 0.958 | 0.950 | 0.954 |
| Pearson r | 0.923 | 0.902 | 0.901 |
| Top 1 MSE | 2.03 | 2.06 | 2.05 |
| Top 50 MSE | 2.24 | 2.30 | 2.32 |

All models have excellent local structure (r > 0.95). Differences are negligible.

### Final Recommendation: Beta=0.05

**Best overall configuration: Latent=384, Beta=0.05**

| Criterion | Beta=0.05 |
|-----------|-----------|
| Reconstruction MSE | 1.030 (best) |
| 6mer MSE | 1.319 (best) |
| Training stability | Stable |
| Local structure (Spearman r) | 0.950 (excellent) |
| KL divergence | 1261 (healthy, expressive) |

Beta=0.05 provides the best balance of reconstruction quality, training stability, and local structure preservation for vector DB retrieval.

---

## 2025-12-03 ~13:00: Reset beta to 0.05 in VAEMulti.py

Reverted beta from 0.03 back to 0.05 in VAEMulti.py to match the optimal configuration determined through systematic testing. The code now uses `max_weight = 0.05` in the KLWarmupCallback.

---

## 2025-12-03 ~14:15: Updated for new k-mer file format

### Changes

Updated column indices to match new `calculate_kmer_frequencies` output format (without 7-mers):

**Old format (all_kmers.npy with 7-mers):**
- Column 0: row index
- Columns 1-8192: 7-mers
- Columns 8193-10964: 6-mers through 1-mers

**New format (k-mers.npy without 7-mers):**
- Column 0: sequence length
- Columns 1-2772: 6-mers through 1-mers

Code changes:
```python
# Old
COL_START = 8193
COL_END = 10965

# New
COL_START = 1
COL_END = 2773
```

---

## 2025-12-03 ~19:00: Distance distribution analysis

### Observation

When analyzing pairwise distances in the latent space:
- **Euclidean distance**: Unimodal distribution
- **Cosine distance**: Bimodal distribution

### Interpretation

**Euclidean distance** measures absolute distance in the 384-dimensional space. In high dimensions, distances tend to concentrate around a mean value (the "curse of dimensionality") - most points end up roughly the same distance apart, giving a unimodal distribution.

**Cosine distance** measures the angle between vectors, ignoring magnitude. This is sensitive to the *direction* of vectors in the latent space. The bimodal distribution suggests there are two dominant "directions" or clusters in the latent space - possibly:
- Chromosomal DNA vs. mobile elements (plasmids/viruses)
- Bacteria vs. archaea
- High GC vs. low GC organisms
- Or some other fundamental compositional split in the data

### Significance

This is a good sign - it means the VAE is learning meaningful structure that separates major groups. The bimodality in cosine distance is why ChromaDB is configured with `'hnsw:space': 'cosine'` - it should be better at distinguishing these groups for retrieval.

---

## 2026-01-31: Clustering notebook updates and CLAUDE.md sync

### Changes to clustering.ipynb
- Split data loading cell from plotting cell (faster iteration on plots)
- Added `embedding_ids` loading from `Data/all_ids.txt`
- Switched to Seaborn with `darkgrid` style for all plots
- Fixed off-by-one error in closest sequence index calculation
- Changed cell 3 to use random sequence instead of first sequence, now prints ID

### CLAUDE.md updates
Synced CLAUDE.md with current VAEMulti configuration:
- Latent dim: 384 (was 256)
- β: 0.05 (was 0.1)
- Input features: 2,772 (6-mers through 1-mers, no GC column)
- Updated file references from VAE.py to VAEMulti.py
- Added instruction to read VAEMulti.md at start of each conversation

### Observation
Minimum cosine distance between sequences is ~0.4 - no very close neighbors found yet. The bimodal distribution persists across random query sequences.

### Nearest neighbor analysis (100k sample)

Sampled 100,000 sequences and computed pairwise cosine distances to find nearest neighbors:

| Metric | Value |
|--------|-------|
| Sequences with neighbor < 0.1 | 1,048 (1.05%) |
| Minimum distance found | 0.0005 |
| Mean nearest neighbor distance | 0.5216 |

**Key findings:**
- Close neighbors DO exist, but are rare (~1% within 0.1 distance)
- Near-duplicates exist (0.0005 ≈ identical) - likely same organism or closely related strains
- Most sequences are spread out (mean NN distance 0.52)
- The latent space is not collapsing - embeddings maintain meaningful distances

### Count vs distance analysis

Binned nearest neighbor distances and plotted count vs distance to examine the geometry of the latent space.

| Distance Range | R² (linear fit) | Slope | Intercept |
|----------------|-----------------|-------|-----------|
| 0 - 0.2 | 0.917 | 439.2 | -1.2 |
| 0 - 0.5 | 0.954 | 1644.7 | -70.2 |
| 0 - 0.7 | 0.677 | 4196.2 | -469.6 |

**Interpretation:**
- **Linear regime (0 - 0.5)**: Count grows linearly with distance (R² > 0.95). This is surprising for a 384-dim space where volume should grow as r^383. Suggests embeddings lie on a low-dimensional manifold.
- **Transition (~0.5 - 0.7)**: Supra-linear growth begins, R² drops. Starting to see higher-dimensional structure.
- **Bimodal connection**: The transition at ~0.5-0.7 aligns with the bimodal cosine distance distribution observed earlier - this is where the two modes meet.

**Implication**: The VAE has learned a structured, low-dimensional representation. Local distances (< 0.5) behave as if on a low-dim manifold, which is good for nearest-neighbor retrieval.

---

## 2026-02-01: t-SNE visualization (full dataset)

### Setup
- Used openTSNE library (fast, supports millions of points)
- Ran on full 4,776,770 embeddings
- Parameters: perplexity=30, metric='cosine', n_jobs=-1
- Memory usage: ~75% of 512 GB
- Time: ~25 minutes on 32-core Threadripper

### Results

The t-SNE visualization reveals clear structure dominated by GC content:

**Two major lobes:**
- Left lobe (blue): Low GC organisms (~20-40%)
- Right lobe (red/orange): High GC organisms (~50-70%)

**Key findings:**
1. The bimodal cosine distance distribution is explained by GC content split
2. Clear gradient from low to high GC across the visualization
3. Satellite clusters/islands around edges (possibly specific taxa or mobile elements)
4. Transition zone in middle shows intermediate GC sequences

**Biological interpretation:**
GC content is the primary axis of variation in the latent space, which makes sense - it's one of the strongest signals in k-mer frequencies. The VAE has learned compositionally meaningful structure.

### Memory scaling observation

Testing memory usage with different sample sizes:
| Sample Size | Memory Usage |
|-------------|--------------|
| 1M | ~45% (230 GB) |
| 2M | ~53% (272 GB) |
| 4.8M (full) | ~75% (384 GB) |

Memory scaling is sublinear - significant fixed overhead from Annoy index and FFT structures.
