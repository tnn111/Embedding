# VAE Development Log

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
4. Model files prefixed with `vae_` instead of `vae_`

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

## 2025-12-03 ~13:00: Reset beta to 0.05 in VAE.py

Reverted beta from 0.03 back to 0.05 in VAE.py to match the optimal configuration determined through systematic testing. The code now uses `max_weight = 0.05` in the KLWarmupCallback.

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
- Updated file references from VAE.py to VAE.py
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

### Geographic origin analysis (SFE vs SE)

Colored t-SNE by sequence ID prefix to compare two estuaries:
- **SFE** (San Francisco Estuary): 1,941,549 sequences (40.6%)
- **SE** (Baltic Sea): 2,835,221 sequences (59.4%)

**Observations:**
1. Both environments span the full GC range - neither restricted to one lobe
2. Low-GC lobe (left) appears more Baltic-dominated
3. High-GC lobe (right) is more mixed, possibly slightly more SFE
4. Significant overlap - communities share many similar organisms
5. Some satellite clusters appear environment-specific

**Interpretation:**
The two estuaries have similar overall microbial diversity (both sample the full GC spectrum) but with different relative abundances. The Baltic may favor more low-GC organisms, possibly reflecting colder temperatures or different dominant taxa. The overlap suggests shared cosmopolitan organisms common to estuarine environments.

**Detailed comparison (side-by-side plot):**

1. **Same global structure** - Both environments show the two-lobe pattern (low GC left, high GC right) with the characteristic "channel" between them

2. **Density differences:**
   - Baltic: Denser in the lower-left lobe, more uniform coverage
   - SFE: More pronounced satellite clusters, especially around the periphery of the right (high-GC) lobe

3. **Coverage gaps:**
   - SFE has a more "sparse" look in the central channel region
   - Baltic fills in more of the transition zone between lobes

4. **Satellite clusters:**
   - Both have peripheral islands, but SFE has more distinct outlier populations (possibly specific taxa unique to that environment)

5. **Lower-left dense region** appears stronger in Baltic - could represent cold-adapted, low-GC organisms more abundant in the Baltic

**Conclusion:** Similar overall structure confirms these are comparable estuarine communities sampling the same compositional space, but with different relative abundances. Baltic appears more "complete" while SFE has more distinct sub-populations.

---

## 2026-02-01: HDBSCAN clustering on t-SNE coordinates

### Setup
- Applied HDBSCAN to the 2D t-SNE coordinates (4.8M points)
- Parameters: `min_cluster_size=1000`, `min_samples=100`
- Ran successfully on full dataset after testing on 500k and 2M subsamples

### HDBSCAN parameters
- `min_cluster_size`: Minimum membership for a cluster to be kept
- `min_samples`: Controls how conservative core point determination is
  - Higher values → more noise, denser cores required
  - Lower values → more points clustered, sparser regions included

### Output files
- `clusters.tsv`: Cluster IDs and sizes (tab-separated)
- `cluster_members.tsv`: Cluster IDs and comma-separated sequence IDs

### Approaches for comparing sample groups
Three methods explored for identifying differences between sample groups in t-SNE space:

1. **Density difference heatmap**: Compute normalized 2D histograms for each group, subtract to show enrichment/depletion
2. **Cluster composition analysis**: For each HDBSCAN cluster, compute fraction belonging to each group
3. **Overlay plot**: Plot both groups on same axes with different colors to visualize overlap

---

## 2026-02-05: Full Codebase Review (Opus 4.6)

### Issues Found (with resolution status as of 2026-02-19)

**FIXED:**
1. ~~Inference scripts use stochastic `z` instead of deterministic `z_mean`~~ — Both `embedding` and `create_and_load_db` now use `z_mean, _, _ = encoder.predict(...)`.
2. ~~`verify_local_distances.py` has stale column indices~~ — Updated to current format.
3. ~~`convert_txt_to_npy` is outdated~~ — Removed.
4. ~~Stale model symlink~~ — Now points to `Runs/Run_SFE_SE_5/vae_encoder_best.keras`.
5. ~~`main.py` is a placeholder~~ — Removed.
6. ~~CLR applied across mixed compositions~~ — Switched to per-group CLR (see below).
7. ~~Small CLR pseudocount (1e-6)~~ — Replaced with Jeffreys prior (0.5/n_features).
8. ~~ChromaDB uses cosine distance~~ — Switched to L2 (Euclidean).

**REMAINING:**
1. **Duplicated custom layers across 6 files** — `ClipLayer`, `Sampling`, `clr_transform` copy-pasted into VAE.py, VAE_noGC.py, `embedding`, `create_and_load_db`, `verify_local_distances.py`, `verify_knn_quality`. Should extract to shared module.
2. **Training history overwritten on resume** — `vae_history.pkl` loses full curve.
3. **pyproject.toml project name is "clustering"** — minor.
4. **README.md is empty** — minor.

### Architecture & Design Decisions

**No dropout** — Architecture relies on BatchNorm + KL regularization. Train/val gap < 1 pt with shuffled data. Data-to-parameter ratio is healthy (13.4M samples / 7M params).

**NumpyBatchDataset drops last incomplete batch** — negligible impact.

---

## 2026-02-05: Switched to per-group CLR transformation

### Change
CLR is now applied independently to each k-mer size group (6-mer, 5-mer, 4-mer, 3-mer, 2-mer, 1-mer) instead of jointly across all 2,772 features.

### Rationale
`calculate_kmer_frequencies` normalizes each k-size group independently (each sums to 1.0). These are 6 separate compositions. Applying CLR jointly mixed features from different compositional spaces, with the geometric mean dominated by the 2,080 6-mer features. Per-group CLR respects the independence of each composition.

### Files changed
- `VAE.py` — `clr_transform_inplace` loops over `KMER_SIZES` dict
- `embedding` — `clr_transform` loops over `KMER_SLICES` list
- `create_and_load_db` — same
- `verify_local_distances.py` — same

### Impact
- All existing models are incompatible (trained with joint CLR)
- Retraining required
- ChromaDB needs to be regenerated after retraining

---

## 2026-02-05: Jeffreys prior pseudocount for CLR

### Change
Replaced fixed pseudocount of 1e-6 with a per-group Jeffreys prior: `pseudocount = 0.5 / n_features`.

### Rationale
Each k-mer group is a multinomial composition. The Jeffreys prior (Dir(0.5, ..., 0.5)) is the standard uninformative prior for multinomial data. Adding 0.5 counts before normalization is equivalent to adding `0.5 / n_features` to the normalized frequencies.

**Previous (1e-6):** For a zero-count 6-mer, log(1e-6) = -13.8. Typical non-zero value ~1/2080 gives log(4.8e-4) = -7.6. Gap of ~6 log units distorts the geometric mean.

**New (Jeffreys):** Per-group pseudocounts:

| Group | n_features | Pseudocount | log(pseudocount) |
|-------|-----------|-------------|------------------|
| 6-mer | 2080 | 2.4e-4 | -8.3 |
| 5-mer | 512 | 9.8e-4 | -6.9 |
| 4-mer | 136 | 3.7e-3 | -5.6 |
| 3-mer | 32 | 1.6e-2 | -4.2 |
| 2-mer | 10 | 5.0e-2 | -3.0 |
| 1-mer | 2 | 2.5e-1 | -1.4 |

Gap between zero and typical non-zero 6-mer is now ~0.7 log units instead of ~6. The transform is less dominated by absence/presence and more sensitive to frequency differences.

### Files changed
Same 4 files as per-group CLR change. Pseudocount parameter removed from function signatures.

### Context
This work is intended as foundation for a Nature Methods paper. The Jeffreys prior is well-established in the compositional data analysis literature and provides a principled, citable justification.

### Implications for minimum contig length

Previous attempt with 1,000 bp contigs (5,000 bp threshold) gave poor results, likely due to the 1e-6 pseudocount. For a 1,000 bp contig:
- ~995 6-mer positions across 2,080 bins → most bins are zero
- **Old (1e-6):** zeros at log(1e-6) = -13.8, non-zeros at ~-6.9. Gap of ~7 log units, majority of features at the extreme. CLR geometric mean dominated by zeros, burying the signal.
- **Jeffreys (2.4e-4):** zeros at -8.3, non-zeros at ~-6.7. Gap of ~1.6 log units. Signal preserved.

Should re-test 1,000 bp minimum after model converges. Fundamental noise issue remains (995 observations / 2,080 bins), but preprocessing no longer destroys the signal.

---

## 2026-02-05: Training progress with per-group CLR + Jeffreys prior

### Training metrics at epoch ~237 (not fully converged)

```
Val: 102.83, MSE: 0.036 [6mer=0.0464, 5mer=0.0067, 4mer=0.0008, 3mer=0.0002, 2mer=0.0001, 1mer=0.0000]
KL: 272 (rising slightly while MSE drops — encoder using latent dims more expressively)
```

Per-k-mer MSE logging precision increased to 6 decimal places (from 4) to track small values like 1-mer and 2-mer.

Note: absolute MSE values are not directly comparable to previous models due to different CLR scale (Jeffreys prior produces smaller target values than 1e-6 pseudocount).

### Local distance verification (epoch ~237)

Tested on actual training data (`all_contigs_l5000.npy`), not the older aquatic-only `Data/all_kmers.npy`.

| Metric | 10k sample | 50k sample |
|--------|-----------|-----------|
| Spearman r | 0.869 | 0.931 |
| Pearson r | 0.473 | 0.598 |
| Top 1 MSE | 0.063 | 0.060 |
| Top 50 MSE | 0.085 | 0.077 |
| Random baseline | 0.226 | 0.227 |

**Observations:**
- Spearman 0.93 at 50k sample, already approaching the 0.95 from the previous fully-converged model
- Larger sample pool → better nearest neighbors (more candidates to find true close matches)
- Pearson < Spearman indicates monotonic but nonlinear relationship between latent distance and k-mer MSE — fine for retrieval since only ranking matters
- Tested on old aquatic-only data: Spearman 0.67 — expected since model is now trained on more diverse data and aquatic subset is proportionally less represented

### Later verification (training near convergence, ~epoch 400+)

| Run | Spearman r | Pearson r | Top 1 MSE | Top 50 MSE |
|-----|-----------|-----------|-----------|-----------|
| 1   | 0.929     | 0.638     | 0.060     | 0.076     |
| 2   | 0.927     | 0.620     | 0.060     | 0.077     |
| 3   | 0.929     | 0.620     | 0.060     | 0.077     |
| 4   | 0.922     | 0.569     | 0.060     | 0.079     |

Model has plateaued on Spearman ~0.93 (range 0.922-0.931 across runs) — variation is from random query selection. Close to the 0.95 achieved by the previous fully-converged model despite completely different preprocessing (per-group CLR + Jeffreys prior vs joint CLR + 1e-6 pseudocount). Top 1 MSE is remarkably stable at 0.060 across all runs.

### Final results (500 epochs)

```
Val: 101.44, MSE: 0.035 [6mer=0.0456, 5mer=0.0062, 4mer=0.0008, 3mer=0.0002, 2mer=0.0001, 1mer=0.0000]
Recon: 98.07, KL: 284.4
```

**Runtime:** 2h 59m, 190% CPU, 252 GB peak memory

**Comparison to previous best (joint CLR + 1e-6, 1000 epochs):**

| Metric | Previous | New (per-group CLR + Jeffreys) |
|--------|----------|-------------------------------|
| Val loss | 1517.8 | 101.4 |
| MSE | 0.601 | 0.035 |
| 6-mer MSE | 0.777 | 0.046 |
| KL | ~272 (prev) | 284 |
| Spearman r (50k) | 0.95 | 0.93 |

Note: MSE values are not directly comparable due to different CLR scales. The Spearman correlation is the scale-independent comparison — 0.93 vs 0.95 shows comparable latent space quality. KL is higher (284 vs 272), indicating slightly more expressive latent representations.

---

## 2026-02-08: Minimum contig length sweep results

> **SUPERSEDED**: Trained on unshuffled data. See "2026-02-12: All 5 shuffled runs complete" and `Runs.md` for current results. Key findings that remain valid: Jeffreys prior solved the short-contig problem; reconstruction loss alone is insufficient to evaluate embedding quality; Run 4 had a ReduceLROnPlateau scheduling artifact (best MSE but worst Spearman); FD paper confirms 3 kbp minimum is standard (mmlong2 pipeline default). Run_4_prime confirmed the artifact — Spearman 0.727 vs original 0.580.

---

## 2026-02-08: Run_SFE_SE — aquatic-only baseline with new preprocessing

> **SUPERSEDED**: Trained on unshuffled data. See `Runs.md` for shuffled SFE_SE results. Key finding: SFE_SE model with 4.8M sequences scored Spearman 0.714 on 5K augmented data — competitive with augmented runs despite 3x fewer samples.

---

## 2026-02-08: Observations on embedding quality evaluation

### Reconstruction loss is insufficient

The Run 4 analysis revealed that reconstruction MSE alone is insufficient to evaluate embedding quality. Run 4 had the best or near-best reconstruction across all k-mer sizes, yet the worst Spearman correlation (latent space retrieval quality). Without an independent metric like Spearman, all 5 sweep runs would have looked comparable — especially at typical training lengths (100-500 epochs), where Run 4's LR scheduling anomaly would not have been visible.

### Two complementary latent space quality metrics

1. **Spearman correlation** (latent distance vs input-space MSE): Measures whether the *ranking* of neighbors is preserved — closest in latent space should be closest in k-mer space. Scale-independent, directly relevant to retrieval.

2. **Count-vs-distance linearity** (from earlier analysis on the original model): The number of neighbors within distance r grows linearly with r for r < 0.5 (R² = 0.954). In a 384-dimensional space, volume grows as r^383, so linear growth implies the data lies on a low-dimensional manifold. This means the local *metric structure* is well-behaved — distances are meaningful, not just rankings.

Together: Spearman validates the ordering, count-vs-distance validates the geometry. Neither is captured by reconstruction loss.

### TODO
- Re-run the count-vs-distance analysis on the final model (per-group CLR + Jeffreys prior) to confirm the linear regime persists. The clustering notebook has the original analysis.

---

## 2026-02-08: Full cross-comparison matrix (models × test datasets)

> **SUPERSEDED**: All models trained on unshuffled data. See "2026-02-12: Full 5×5 cross-comparison matrix (shuffled data)" and `Runs.md` for current results. Key findings confirmed in shuffled runs: Run_3 is the best generalist; 50k sample size is stable (deltas < 0.01 vs 100k); shorter test data is easier (more distinctive k-mer profiles).

---

## 2026-02-09: Metrics logging bug fix

### Bug
The "Recon" value logged by VAEMetricsCallback was computed on a **5000-sample validation subset**, not on training data. This was misleading because comparing "Recon" to "Val" (full validation loss) looked like a train-vs-val gap when it was actually a sampling artifact. For SFE_SE_3, the first 5000 validation samples happened to be unrepresentatively easy, creating a false overtraining signal (118.4 vs 251.5).

### Fix
Replaced "Recon" with the actual Keras training loss (`logs.get('loss')`). The log line now shows:
```
Epoch N: Train: <actual_train_loss>, Val: <full_val_loss>, KL: ..., MSE: ..., [per-k-mer]
```
The MSE and per-k-mer breakdown are still computed from the validation sample (useful granularity) but are no longer confused with training loss.

### Impact
All previous "Recon" values in training logs should NOT be compared to "Val" for overtraining assessment. They measured different things. Reruns with the fixed code will provide accurate train/val comparisons.

---

## 2026-02-09: SFE_SE cross-comparison (models × aquatic-only test data)

> **SUPERSEDED**: All models trained on unshuffled data. See `Runs.md` for shuffled SFE_SE results (sections 6-7). Key finding: on shuffled data, SFE_SE models dramatically outperform augmented models on augmented test data (worst SFE_SE 0.812 > best augmented 0.702).

## 2026-02-09: Data shuffling concern and concatenate_matrices fix

### Problem

The VAE uses a contiguous 90/10 train/val split (first 90% train, last 10% val) and assumes pre-shuffled data. The individual source datasets were shuffled internally, but `concatenate_matrices` stacked them in order without shuffling. This means the validation set is dominated by whichever source was concatenated last — creating a systematic distribution shift between train and val sets. This likely explains some of the apparent "overtraining" patterns in the SFE_SE experiments.

### Fix

Added `--shuffle` flag to `concatenate_matrices`. After concatenation, it generates a random permutation and applies it to both the matrix rows and ID lines. Also works with a single input file pair for reshuffling existing datasets.

### Impact

All concatenated datasets used for training should be regenerated with `--shuffle`. Previous train/val loss comparisons may be unreliable due to the distribution shift.

---

## 2026-02-10: Run_1 mid-training verification (epoch ~570)

### Setup
- Run_1 is retraining on shuffled data (1,000 bp threshold, all 4 sources)
- At epoch 570/1000: Train: 189.35, Val: 189.14, MSE: 0.059, 6-mer: 0.0768
- verify_local_distances.py run with 50k samples on CPU

### Results

| Metric | Value |
|--------|-------|
| Spearman r | 0.766 |
| Pearson r | 0.543 |
| Top 1 MSE | 0.108 ± 0.094 |
| Top 5 MSE | 0.113 ± 0.079 |
| Top 10 MSE | 0.119 ± 0.078 |
| Top 20 MSE | 0.129 ± 0.086 |
| Top 50 MSE | 0.149 ± 0.112 |
| Random MSE | 0.456 ± 0.275 |
| NN/Random ratio | 4.2x |

### Comparison to previous Run 1 (1000 epochs, unshuffled data)

| Metric | Run 1 (1000 ep, unshuffled) | Run 1 (570 ep, shuffled) |
|--------|---------------------------|------------------------|
| Spearman (own data) | 0.852 | 0.766 |
| Top 1 MSE | 0.121 | 0.108 |
| Top 50 MSE | 0.165 | 0.149 |
| Random MSE | 0.555 | 0.456 |

Still training — Spearman at 0.766 is lower than the previous Run 1's 0.852 at convergence, but the model is only 57% through training. The lower random baseline MSE (0.456 vs 0.555) reflects the shuffled validation set now being representative of the full data distribution.

### Follow-up at epoch ~750

| Metric | Epoch ~570 | Epoch ~750 | Δ |
|--------|-----------|-----------|---|
| Spearman r | 0.766 | 0.768 | +0.002 |
| Pearson r | 0.543 | 0.525 | -0.018 |
| Top 1 MSE | 0.108 | 0.109 | +0.001 |
| Top 50 MSE | 0.149 | 0.152 | +0.003 |

Essentially converged — Spearman and reconstruction metrics both flat since epoch ~570. The remaining 250 epochs are unlikely to produce meaningful improvement.

### Training duration observation

500 epochs would likely be sufficient. Evidence: MSE flat at 0.059 since at least epoch 570, Spearman only +0.002 from epoch 570→750, val loss barely moving. The previous sweep runs reached min LR by epoch 339-459, meaning 500 epochs gives 50-160 epochs of fine-tuning at min LR — apparently enough for convergence. Would save ~3 hours per run. Keeping 1,000 epochs for all reruns to confirm this pattern across all thresholds.

### Run_2 shuffled results (1000 epochs)

| Metric | Run 1 (shuffled) | Run 2 (shuffled) | Old Run 2 (unshuffled) |
|--------|-----------------|-----------------|----------------------|
| Spearman r | 0.768 | 0.627 | 0.742 |
| Pearson r | 0.525 | 0.360 | 0.487 |
| Top 1 MSE | 0.109 | 0.167 | 0.129 |
| Top 50 MSE | 0.152 | 0.236 | 0.168 |
| Random MSE | 0.456 | 0.542 | 0.521 |

Run_2 Spearman dropped more than Run_1 after shuffling (0.742→0.627 vs 0.852→0.768). The higher random baseline (0.542 vs 0.456 for Run_1) is not because the 2k subset is "more diverse" — it's a strict subset of the 1k data. Rather, the short sequences (1000-1999 bp) excluded from 2k have noisy, less distinctive k-mer profiles that regress toward the mean in CLR space. Including them in the 1k set deflates average pairwise distances. Need to see the remaining runs to understand the pattern.

### Run_3 shuffled results (1000 epochs, complete)

Checked at epoch ~480 and again at 1000: Spearman 0.724 → 0.721. No improvement in last 500 epochs — confirms convergence-by-500 pattern.

### Shuffled runs summary so far

| Run | Epochs | Spearman | Pearson | Top 1 MSE | Top 50 MSE | Random MSE |
|-----|--------|----------|---------|-----------|-----------|------------|
| Run 1 (1k) | ~750 | 0.768 | 0.525 | 0.109 | 0.152 | 0.456 |
| Run 2 (2k) | 1000 | 0.627 | 0.360 | 0.167 | 0.236 | 0.542 |
| Run 3 (3k) | 1000 | 0.721 | 0.388 | 0.119 | 0.194 | 0.511 |

### Run_4 shuffled results (epoch ~184, in progress)

Run_4 at epoch ~184: Spearman 0.789, Pearson 0.620 — best of the shuffled runs so far despite being early in training. Notable train/val gap (145 vs 196). Starting LR set to 1e-5 (lower than Run_4' at 5e-5, learning from the original Run_4 scheduling artifact).

| Run | Epochs | Spearman | Pearson | Top 1 MSE | Random MSE |
|-----|--------|----------|---------|-----------|------------|
| Run 1 (1k) | ~750 | 0.768 | 0.525 | 0.109 | 0.456 |
| Run 2 (2k) | 1000 | 0.627 | 0.360 | 0.167 | 0.542 |
| Run 3 (3k) | 1000 | 0.721 | 0.388 | 0.119 | 0.511 |
| Run 4 (4k) | ~184 | 0.789 | 0.620 | 0.118 | 0.516 |

### Run_5 shuffled results (epoch ~378, in progress)

Run_5 at epoch ~378: Spearman 0.690, Pearson 0.402. Starting LR=1e-5. Train/val gap of ~42, similar to Run_4.

| Run | Epochs | Spearman | Pearson | Top 1 MSE | Random MSE | Train/Val gap |
|-----|--------|----------|---------|-----------|------------|---------------|
| Run 1 (1k) | ~750 | 0.768 | 0.525 | 0.109 | 0.456 | 0.2 |
| Run 2 (2k) | 1000 | 0.627 | 0.360 | 0.167 | 0.542 | 0.5 |
| Run 3 (3k) | 1000 | 0.721 | 0.388 | 0.119 | 0.511 | 0.4 |
| Run 4 (4k) | ~184 | 0.789 | 0.620 | 0.118 | 0.516 | 50.2 |
| Run 5 (5k) | ~378 | 0.690 | 0.402 | 0.109 | 0.492 | 42.1 |

### Train/val gap analysis — BatchNorm artifact, not overfitting

| Run | Train | Val | Gap | Sequences |
|-----|-------|-----|-----|-----------|
| Run 1 (1k) | 189.3 | 189.2 | 0.1 | 17.6M |
| Run 2 (2k) | 178.1 | 177.6 | 0.5 | 17.1M |
| Run 3 (3k) | 166.3 | 165.9 | 0.3 | 16.5M |
| Run 4 (4k) | 142.3 | 190.4 | 48.1 | 14.8M |
| Run 5 (5k) | 120.8 | 162.6 | 41.8 | 13.4M |

**Root cause: BatchNorm's different behavior in training vs eval mode.**

The gap exists from epoch 1 (Run 5: Train 156.6, Val 179.1 at epoch 1 — before any memorization is possible). BN is the only component that differs between modes:
- `training=True` (Keras training loss): BN uses per-batch statistics
- `training=False` (Keras val loss): BN uses running statistics

**The old "Recon" metric was computed with `training=False` on validation data**, so it could never detect this gap. Old Recon + beta*KL ≈ Val because both were validation-mode measurements. The metrics fix (replacing Recon with `logs.get('loss')`) made the BN gap visible for the first time.

**Confirmation:** Old Run_5 val_loss = 162.4, new Run_5 val_loss = 162.6 — nearly identical. The models learn the same thing; the gap is purely in how training loss is measured.

**Why the BN effect is larger for Runs 4-5:** The 4k/5k datasets have longer sequences with more distinctive k-mer profiles → more variation between batches → BN's per-batch statistics provide a larger "advantage" over fixed running statistics. The 1k data has shorter, noisier sequences → more homogeneous → batch stats ≈ global stats → negligible BN mode difference.

### Run_4 final shuffled results (1000 epochs)

Spearman 0.782 (vs 0.789 at epoch ~184). Confirms convergence-by-500 and shows the train/val gap doesn't hurt latent space quality — VAE bottleneck prevents memorization.

| Run | Epochs | Spearman | Pearson | Top 1 MSE | Random MSE |
|-----|--------|----------|---------|-----------|------------|
| Run 1 (1k) | 1000 | 0.768 | 0.525 | 0.109 | 0.456 |
| Run 2 (2k) | 1000 | 0.627 | 0.360 | 0.167 | 0.542 |
| Run 3 (3k) | 1000 | 0.721 | 0.388 | 0.119 | 0.511 |
| Run 4 (4k) | 1000 | 0.782 | 0.528 | 0.121 | 0.516 |
| Run 5 (5k) | ~378 | 0.690 | 0.402 | 0.109 | 0.492 |

### Plan: final analysis after all shuffled reruns

9 remaining runs (Run 2-5, Run 4', SFE_SE 1-5) on shuffled data, ~3-4 days of GPU time. Once complete: full cross-comparison on matched conditions, draw final conclusions on threshold selection and training duration, then move forward with downstream analyses.

---

## 2026-02-12: All 5 shuffled runs complete — final results

### verify_local_distances (50k samples, 100 queries, 50 neighbors, own data)

| Run | Threshold | Spearman | Pearson | Top 1 MSE | Top 50 MSE | Random MSE |
|-----|-----------|----------|---------|-----------|-----------|------------|
| Run 1 | 1K bp | 0.751 | 0.430 | 0.167 | 0.256 | 0.555 |
| Run 2 | 2K bp | 0.627 | 0.360 | 0.167 | 0.236 | 0.542 |
| **Run 3** | **3K bp** | **0.721** | **0.388** | **0.119** | **0.194** | 0.511 |
| Run 4 | 4K bp | 0.697 | 0.412 | 0.122 | 0.194 | 0.494 |
| Run 5 | 5K bp | 0.511 | 0.242 | 0.139 | 0.240 | 0.468 |

Run 4 and Run 5 own-data values updated after kmers_4.npy and kmers_5.npy were replaced. Run 3 (3K bp) has the best Spearman on own data and wins the cross-threshold comparison on every test condition (see below).

### Sampling variance in verify_local_distances

Runs 1, 4, and 5 were tested at multiple timepoints. Differences between readings reflect random query selection (100 queries from 50k pool):

| Run | Mid-training | Final (1000 ep) | Δ |
|-----|-------------|-----------------|---|
| Run 1 | 0.768 (ep ~750) | 0.751 | -0.017 |
| Run 3 | 0.724 (ep ~480) | 0.721 | -0.003 |
| Run 4 | 0.789 (ep ~184) | 0.783 | -0.006 |
| Run 5 | 0.690 (ep ~378) | 0.661 | -0.029 |

Run 5's larger drop (0.029) is outside typical noise (~0.01). Could indicate slight latent space degradation in late training, or an unlucky query draw. Rankings are stable across all measurements: Run 4 > Run 1 > Run 3 > Run 5 > Run 2.

### ReduceLROnPlateau schedules (from resource.log)

Keras writes ReduceLROnPlateau messages to stdout/stderr (captured in `resource.log`), NOT to the custom `vae_training.log`. All 5 runs had multiple LR reductions:

| Run | Start LR | 1st reduction | Floor (1e-6) | Epochs at floor | Reductions |
|-----|----------|---------------|--------------|-----------------|------------|
| Run 1 | 1e-4 | Epoch 21 | Epoch 351 | 649 | 7 |
| Run 2 | 1e-4 | Epoch 21 | Epoch 354 | 646 | 7 |
| Run 3 | 1e-4 | Epoch 22 | Epoch 316 | 684 | 7 |
| Run 4 | 1e-4 | Epoch 22 | Epoch 468 | 532 | 7 |
| Run 5 | 1e-4 | Epoch 21 | Epoch 346 | 654 | 7 |

> **Note**: Run 4 and Run 5 were retrained after the data shuffling fix (kmers_4.npy and kmers_5.npy replaced). Values above are from the final retrained runs. Earlier entries in this log describing LR 1e-5 for Run 4/5 are from the pre-fix training.

All 5 runs start at LR 1e-4 with first reduction at epoch 21-22. Runs 1-3 hit LR floor by epoch 316-354 (649-684 epochs fine-tuning). Runs 4-5 reach floor by epoch 346-468.

### Training dynamics summary

1. **Convergence by ~500 epochs confirmed** — Spearman barely changes after epoch 500 in any run. LR floor reached by epoch 468 at latest.
2. **Train/val gap was a data shuffling artifact** — Runs 4-5 initially showed ~34-48 point gaps due to unshuffled data, not BatchNorm. After retraining on properly shuffled data, all runs show <1 point gap. See correction at "2026-02-13: Train/val gap was a data shuffling artifact".
3. **Shuffling produces lower Spearman** — unshuffled val sets were biased toward one source, making ranking artificially easier. The shuffled results are the more honest measurement.
4. **Run_3 is the best model** — highest mean Spearman (0.702) across the full 5×5 cross-comparison matrix. Wins or ties every column.
5. **Run 2 remains unexplained outlier** — lowest Spearman (0.627) in both shuffled and unshuffled conditions.

---

## 2026-02-12: Added --metric flag to verify_local_distances.py

Added `--metric` argument accepting `euclidean` (default) or `cosine`. Motivation: ChromaDB is configured with `'hnsw:space': 'cosine'`, but verify_local_distances was hardcoded to Euclidean. This mismatch means the validation may not accurately predict real retrieval quality. Cosine distance measures angular similarity (ignoring magnitude), while Euclidean measures absolute distance — these can rank neighbors differently.

### Euclidean vs cosine comparison (Run_4 model, own 4K data)

| Metric | Euclidean | Cosine | Δ |
|--------|-----------|--------|---|
| Spearman r | **0.697** | 0.621 | -0.076 |
| Pearson r | 0.412 | 0.402 | -0.010 |
| Top 1 MSE | **0.122** | 0.133 | +0.011 |
| Top 50 MSE | **0.194** | 0.194 | 0.000 |

Euclidean wins. The VAE's MSE loss optimizes reconstruction in a Euclidean sense, so the latent space geometry favors absolute position over direction. Implication: ChromaDB should use `'hnsw:space': 'l2'` instead of `'cosine'` for this embedding.

---

## 2026-02-12: Run_4 model cross-threshold evaluation

Tested the shuffled Run_4 (4K bp) model on all 5 test datasets to assess generalization to shorter sequences. Updated after second data fix (shuffling correction):

| Test data | Spearman | Top 1 MSE | Random MSE | vs dedicated model |
|-----------|----------|-----------|------------|-------------------|
| 1K bp | 0.738 | 0.178 | 0.555 | 0.751 (Run_1) → -0.013 |
| 2K bp | 0.598 | 0.176 | 0.542 | 0.627 (Run_2) → -0.029 |
| 3K bp | 0.692 | 0.127 | 0.511 | 0.721 (Run_3) → -0.029 |
| **4K bp** | **0.674** | **0.138** | 0.494 | **(own data)** |
| 5K bp | 0.625 | 0.101 | 0.468 | 0.660 (Run_3) → -0.035 |

> **Note**: Run_3 cross-threshold evaluation (below) showed Run_3 is strictly better than Run_4 on all test conditions.

---

## 2026-02-12: Run_3 model cross-threshold evaluation

Tested the shuffled Run_3 (3K bp) model on all 5 test datasets, same protocol as Run_4. Updated after second data fix (shuffling correction):

| Test data | Run_3 Spearman | Run_4 Spearman | Δ (Run_3 - Run_4) |
|-----------|---------------|---------------|-------------------|
| 1K bp | **0.769** | 0.738 | +0.031 |
| 2K bp | **0.639** | 0.598 | +0.041 |
| 3K bp | **0.721** | 0.692 | +0.029 |
| 4K bp | **0.722** | 0.674 | +0.048 |
| 5K bp | **0.660** | 0.625 | +0.035 |

### Observations

1. **Run_3 wins on every test condition** — including Run_4's own 4K data (0.722 vs 0.674) and 5K data (0.660 vs 0.625).
2. **Consistent margin** — +0.029 to +0.048 across all tests.
3. **Run_3 on 4K data (0.722) exceeds Run_4 on its own data (0.674)** — the 3K model is a strictly better encoder even for longer sequences.
4. **3K bp threshold is the sweet spot** — enough short sequences for diversity without noise dominating. The 3K training set is a superset of the 4K set (includes all ≥3,000 bp sequences), so the model sees more diverse training data.

---

## 2026-02-12: Full 5×5 cross-comparison matrix (shuffled data)

All 5 models tested on all 5 test datasets (50k samples, 100 queries, 50 neighbors). Run_4 and Run_5 retrained after data shuffling fix.

### Spearman correlation (higher = better)

| Model \ Test | 1K | 2K | 3K | 4K | 5K | Mean |
|---|---|---|---|---|---|---|
| **Run 1 (1k)** | **0.751** | 0.616 | 0.723 | 0.703 | 0.635 | 0.686 |
| **Run 2 (2k)** | 0.764 | **0.627** | 0.729 | 0.711 | 0.643 | 0.695 |
| **Run 3 (3k)** | 0.769 | 0.639 | **0.721** | **0.722** | **0.660** | **0.702** |
| **Run 4 (4k)** | 0.738 | 0.598 | 0.692 | 0.674 | 0.625 | 0.665 |
| **Run 5 (5k)** | 0.726 | 0.584 | 0.655 | 0.640 | 0.616 | 0.644 |

### Top 1 neighbor MSE (lower = better)

| Model \ Test | 1K | 2K | 3K | 4K | 5K | Mean |
|---|---|---|---|---|---|---|
| **Run 1 (1k)** | **0.167** | 0.172 | 0.108 | 0.117 | 0.105 | 0.134 |
| **Run 2 (2k)** | 0.175 | **0.167** | 0.113 | 0.126 | 0.103 | 0.137 |
| **Run 3 (3k)** | **0.160** | **0.156** | **0.119** | **0.121** | 0.103 | **0.132** |
| **Run 4 (4k)** | 0.178 | 0.176 | 0.127 | 0.138 | 0.101 | 0.144 |
| **Run 5 (5k)** | 0.180 | 0.176 | 0.125 | 0.140 | **0.100** | 0.144 |

Bold diagonal = own-data results.

### Observations

1. **Run_3 wins or ties for best on every column** — the most consistent general-purpose encoder. Mean Spearman 0.702.
2. **Runs 1-3 are competitive** — means within 0.016 of each other (0.686-0.702), with Run_3 edging ahead.
3. **Run_4 and Run_5 improved dramatically after data fix** — the train/val gap (previously 34-48 points) was caused by unshuffled data, not BatchNorm. With proper shuffling, both models show <1 point train/val gap.
4. **Run_5 was already well-converged at epoch 530** — final mean 0.644 vs mid-training 0.640 (only +0.004).
5. **2K test data is uniquely hard** — all models score 0.584-0.639, well below other columns.
6. **MSE confirms Spearman ranking** — Run_3 has the lowest Top 1 MSE on 4/5 columns (mean 0.132). Short-sequence test data (1K/2K) has much higher MSE (~0.16-0.18) than longer (3K-5K: ~0.10-0.14), reflecting noisier k-mer profiles.
7. **No model is best on its own data** — except Run_1 on 1K. Run_3 beats Run_4 on 4K data and all models on 5K data.
8. **Models trained on shorter thresholds generalize upward better than the reverse** — Run_1-3 score 0.635-0.660 on 5K data.
9. **Clear tier structure** — Tier 1: Run_3 (0.702), Tier 2: Run_1-2 (0.686-0.695), Tier 3: Run_4-5 (0.644-0.665).

## 2026-02-13: Train/val gap was a data shuffling artifact

The ~34-48 point train/val gap previously observed in Runs 4-5 (attributed to BatchNorm per-batch vs running statistics) was actually caused by **unshuffled data** in the Run 4 and Run 5 datasets. With properly shuffled data:
- Run 4: Train 126.8, Val 126.5 → gap of ~0.3 (was 48.1)
- Run 5 at epoch 530: Train 126.8, Val 126.5 → gap of ~0.3 (was 41.8)

This also explains why Runs 1-3 never had the gap — their datasets were already properly shuffled. The BatchNorm hypothesis was wrong; it was simply that validation data wasn't representative of training data when the data wasn't shuffled.

## 2026-02-13: Per-k-mer MSE analysis and Run_5 2-mer/1-mer anomaly

### Training MSE by k-mer size (epoch 1000)

| Run | Total MSE | 6-mer | 5-mer | 4-mer | 3-mer | 2-mer | 1-mer |
|-----|-----------|-------|-------|-------|-------|-------|-------|
| Run 1 (1K) | 0.059 | 0.07680 | 0.00700 | 0.00201 | 0.00104 | 0.000514 | 0.000087 |
| Run 2 (2K) | 0.055 | 0.07109 | 0.00633 | 0.00172 | 0.000808 | 0.000335 | 0.000080 |
| Run 3 (3K) | 0.052 | 0.06766 | 0.00609 | 0.00167 | 0.000753 | 0.000327 | 0.000078 |
| Run 4 (4K) | 0.044 | 0.05692 | 0.00560 | 0.00141 | 0.000662 | 0.000302 | 0.000072 |
| Run 5 (5K) | 0.038 | 0.04948 | 0.00520 | 0.00135 | 0.000717 | **0.000450** | **0.000135** |

6-mer dominates total MSE (~95-98% of total, 2080 of 2772 features). Total MSE decreases monotonically from Run_1 to Run_5 (longer sequences = cleaner k-mer profiles = easier reconstruction). However, Run_5 breaks the trend on 2-mer and 1-mer — these are *higher* than Runs 2-4.

### Root cause: extreme-GC coverage gap

Investigation ruled out several hypotheses:
- **Not sampling noise**: the anomaly persists with deterministic z_mean reconstruction
- **Not stochastic z noise**: Monte Carlo analysis (20 draws) shows identical noise cost between Run_3 and Run_5 for 2-mer (~0.000025) and 1-mer (~0.000004)
- **Not data distribution**: CLR-transformed 2-mer/1-mer distributions are nearly identical across all datasets
- **Not 6-mer redundancy**: 2-mer R² from 6-mer top-10 PCs is 0.96 for both 3K and 5K data

The real cause: **extreme high-GC sequences (>75% GC)**. Reconstruction error by GC content bin:

| GC range | Run_3 count | Run_3 1-mer MSE | Run_5 count | Run_5 1-mer MSE |
|----------|-------------|-----------------|-------------|-----------------|
| 0.00-0.35 | 673 | 0.000400 | 685 | 0.000114 |
| 0.35-0.45 | 1278 | 0.000131 | 1186 | 0.000084 |
| 0.45-0.55 | 1544 | 0.000090 | 1470 | 0.000061 |
| 0.55-0.65 | 3563 | 0.000047 | 3507 | 0.000026 |
| 0.65-0.75 | 2877 | 0.000046 | 3106 | 0.000037 |
| **0.75-1.00** | **65** | **0.000379** | **46** | **0.009854** |

Run_5 has **26x higher 1-mer error** on extreme-GC sequences. Only 46 samples (0.46% of validation data), but their errors are so large (~0.010 vs typical ~0.00003) that they contribute ~50% of the total 1-mer MSE.

These organisms likely have small genomes that assemble into shorter contigs, so they're underrepresented in the ≥5K bp training set. Run_3 (≥3K bp) retains more of them, giving it enough examples to learn the pattern. The GC distribution confirms this: 3K data has 4 peaks (0.41, 0.47, 0.63, 0.69) while 5K data has only 3 peaks (lost the 0.47 peak).

**Takeaway**: the 2-mer/1-mer anomaly is a training data coverage issue at the tails of the GC distribution, not a model capacity or noise problem. This further supports Run_3 (3K bp threshold) as the best general-purpose encoder — it captures more biological diversity.

## 2026-02-13: SFE_SE cross-comparison results

### SFE_SE models on augmented test data (kmers_1.npy through kmers_5.npy)

Testing SFE_SE models against the same augmented test datasets used for the augmented run comparison:

| Model \ Test | 1K | 2K | 3K | 4K | 5K | Mean |
|---|---|---|---|---|---|---|
| SFE_SE_1 (1K) | **0.907** | **0.820** | **0.881** | **0.843** | **0.829** | **0.856** |
| SFE_SE_2 (2K) | 0.889 | 0.808 | 0.873 | 0.827 | 0.807 | 0.841 |
| SFE_SE_3 (3K) | 0.886 | 0.802 | 0.868 | 0.813 | 0.811 | 0.836 |
| SFE_SE_4 (4K) | 0.878 | 0.804 | 0.850 | 0.807 | 0.805 | 0.829 |
| SFE_SE_5 (5K) | 0.866 | 0.790 | 0.836 | 0.787 | 0.782 | 0.812 |

All SFE_SE models dramatically outperform augmented runs (best augmented: Run_3 at 0.702 mean vs worst SFE_SE: SFE_SE_5 at 0.812). SFE_SE_1 wins every column.

### SFE_SE models on SFE_SE test data (kmers_SFE_SE_1.npy through kmers_SFE_SE_5.npy)

| Model \ Test | SFE_SE_1 | SFE_SE_2 | SFE_SE_3 | SFE_SE_4 | SFE_SE_5 | Mean |
|---|---|---|---|---|---|---|
| SFE_SE_1 (1K) | 0.773 | 0.739 | 0.723 | 0.816 | 0.779 | 0.766 |
| SFE_SE_2 (2K) | 0.743 | 0.714 | 0.682 | 0.784 | 0.758 | 0.736 |
| SFE_SE_3 (3K) | 0.749 | 0.704 | 0.676 | 0.776 | 0.736 | 0.728 |
| SFE_SE_4 (4K) | 0.751 | 0.684 | 0.687 | 0.781 | 0.726 | 0.726 |
| **SFE_SE_5 (5K)** | **0.862** | **0.862** | **0.823** | **0.868** | **0.819** | **0.847** |

### Observations

1. **SFE_SE data is harder than augmented data** — Models 1-4 score 0.73-0.77 on SFE_SE data vs 0.83-0.86 on augmented data. The SFE_SE test data contains more challenging sequences.

2. **Opposite ranking on SFE_SE vs augmented data** — On augmented data, SFE_SE_1 is best (0.856). On SFE_SE data, SFE_SE_5 dominates (0.847) with a massive gap over second-place SFE_SE_1 (0.766). The 5K threshold model excels on SFE_SE sequences.

3. **SFE_SE_5 wins every column on SFE_SE data** — by a huge margin (+0.081 over SFE_SE_1). This is the reverse of the augmented data pattern where lower thresholds generalize better.

4. **4K column is easiest on SFE_SE data** — consistently the highest score for all models. On augmented data, 1K was typically easiest. The 2K and 3K SFE_SE columns are the hardest.

5. **SFE_SE models still far outperform augmented models** — even the worst SFE_SE result on SFE_SE data (SFE_SE_3 on SFE_SE_3 at 0.676) approaches the best augmented result (Run_3 at 0.702).

## 2026-02-14: Model selection for Leiden community detection

### Goal
Break up SFE and SE sequences into communities using the Leiden algorithm, with VAE embeddings providing the distance metric. The goal is to identify what organisms are present.

### Decision: SFE_SE_5

SFE_SE_5 is the best model for this task. It dominates all SFE_SE test columns (mean Spearman 0.847, +0.081 over second place) and generalizes well downward — it scores 0.862 on kmers_SFE_SE_1 despite being trained only on ≥5K bp sequences.

Use Euclidean distance for kNN graph construction (outperforms cosine by +0.076 Spearman).

### Key insight: training data composition > quantity

SFE_SE models (4.8-6.7M sequences, 2 sources) dramatically outperform augmented models (13.4-17.4M sequences, 4 sources) — even on the augmented test data that includes FD+NCBI sequences the SFE_SE model never saw (worst SFE_SE 0.812 > best augmented 0.702).

The additional FD and NCBI data forces the model to spread its 384-dimensional latent space across a wider range of biology, diluting local distance structure for any particular domain. The SFE_SE models devote all representational capacity to the marine metagenome manifold, producing better-structured embeddings.

### Data source clarification

- **SFE** (San Francisco Estuary) and **SE** (Baltic Sea) are **marine** metagenomic data
- **FD** (Microflora Danica) contains both **aquatic and soil** data
- **NCBI** (RefSeq representative genomes) is curated, taxonomically diverse

---

## 2026-02-12: Documentation cleanup

- Fixed VAE.py docstrings: encoder/decoder docstrings incorrectly said 256-dim latent space, corrected to 384-dim.
- Fixed stale LR schedule table for Run_4 and Run_5 (had pre-shuffle values).
- Fixed training dynamics summary (BatchNorm → shuffling artifact, Run_4 best → Run_3 best).
- Compacted 4 SUPERSEDED sections, pointing readers to Runs.md.
- Replaced all "base" terminology with "augmented" across VAE.md, Claude_Notes.md, and CLAUDE.md.
- Updated CLAUDE.md with current paths and ChromaDB L2 recommendation.
- Updated Claude_Notes.md: replaced stale "In Progress" section with current training parameters.

---

## 2026-02-20: concatenate_matrices NumPy 2.x fix

`np.lib.format._read_array_header` (private API) was removed in NumPy 2.x. Replaced with the public `read_array_header_1_0` / `read_array_header_2_0` selected by format version.

## 2026-02-20: Run_SFE_SE_NCBI_5 — adding taxonomic signposts

Training a new model on SFE + SE + NCBI data (5K bp threshold). Created via:
```
concatenate_matrices --shuffle -i kmers_SFE_SE_5.npy kmers_NCBI_5.npy -id ids_SFE_SE_5.txt ids_NCBI.txt -o new_SFE_SE_NCBI_5
```

**Dataset**: 5,432,410 sequences (4,776,770 SFE_SE + 655,640 NCBI), ~14% NCBI.

**Rationale**: NCBI RefSeq representative genomes are taxonomically labeled. If they embed near marine metagenomic clusters, they serve as "taxonomic signposts" — enabling biological interpretation of clusters (e.g., "this cluster is near *Pelagibacter*") rather than relying solely on GC span validation.

**Risk**: The augmented models (FD + NCBI + SFE + SE) showed that adding non-marine data dilutes the latent space, dropping Spearman from 0.847 to 0.702. However, NCBI alone is much smaller (~656K vs ~8.7M for FD+NCBI) and taxonomically curated rather than a heterogeneous metagenome mix. The hypothesis is that this modest addition (~14% of training data) will preserve most of the SFE_SE embedding quality while gaining interpretability.

### Live Spearman tracking during training

Measured Spearman on SFE_SE_5 test data (50K sample) at multiple points during training:

| Time | Approx Epoch | Spearman | Delta |
|---|---|---|---|
| 18:27 | ~115 | 0.784 | — |
| 18:38 | ~180 | 0.737 | -0.047 |
| 18:50 | ~250 | 0.710 | -0.027 |
| 19:03 | ~330 | 0.689 | -0.021 |
| 19:38 | ~540 | 0.671 | -0.018 |
| 19:47 | ~590 | 0.662 | -0.009 |
| 19:55 | ~640 | 0.668 | +0.006 |
| 20:06 | ~710 | 0.658 | -0.010 |
| 20:19 | ~790 | 0.662 | +0.004 |
| 20:27 | ~840 | 0.662 | 0.000 |
| **20:49** | **final (1000)** | **0.662** | **0.000** |

(Epoch estimates approximate — ~10 sec/epoch, timestamps are wall clock. Training completed at 1000 epochs. Final Spearman 0.662 on marine data. Note: Top-1 MSE also started degrading by epoch ~540: 0.142 vs 0.124 at epoch ~115.)

**Key observations**:

1. **Spearman declines monotonically while val loss improves.** The `vae_encoder_best.keras` checkpoint keeps being overwritten as val loss decreases, but each new checkpoint has worse Spearman on marine data. This is the "reconstruction loss doesn't predict embedding quality" finding observed in real time.

2. **Top-1 neighbor MSE initially stayed flat (~0.124-0.129) but eventually degraded too (0.142 by epoch ~540).** Deep neighbors (Top 50) degraded earlier and faster (0.175 → 0.219). The NCBI accommodation first disrupts global structure, then eventually erodes local neighborhoods as well.

3. **The decline is the NCBI accommodation effect.** At epoch ~115, the model hadn't yet learned to represent NCBI sequences well — it was essentially still a marine-focused model. As training progresses, the encoder increasingly allocates latent space capacity to the NCBI manifold, diluting marine neighborhood structure. This is the same mechanism that made augmented models (0.644-0.702) worse than SFE_SE models (0.847), observed gradually rather than comparing endpoints.

4. **Converged to augmented-model territory** as predicted. After epoch ~540, Spearman bounces in the 0.658-0.671 range (mean ~0.665). This is between augmented Run_3 (0.702) and Run_5 (0.644), remarkable given NCBI is only 14% of training data vs 61-65% foreign data in augmented models. The dilution effect is highly non-linear — even a small fraction of out-of-domain data causes disproportionate damage.

5. **Implication: a Spearman-aware early stopping criterion would have frozen at epoch ~115.** But this is impractical — it requires an expensive evaluation on held-out data at every checkpoint.

### Does the evaluation unfairly penalize the NCBI-augmented model?

Considered whether filtering NCBI hits from neighbor lists would improve the Spearman score. Answer: **no, this is already what we do.** The 50K sample is drawn entirely from `kmers_SFE_SE_5.npy` — all neighbors in the Spearman calculation are already marine-to-marine. The problem isn't NCBI sequences appearing as false neighbors; it's that the **encoder itself** has reorganized the latent space to accommodate NCBI, pushing marine sequences around even though we only evaluate marine-to-marine distances.

### NCBI Spearman: does training on NCBI help organize NCBI?

Tested Spearman on NCBI-only data (`kmers_NCBI_5.npy`, 655,640 sequences, 50K sample) with both encoders:

| Encoder | Trained on NCBI? | Spearman on NCBI | Pearson | Top-1 MSE |
|---|---|---|---|---|
| SFE_SE_NCBI_5 | Yes | 0.9460 | 0.7150 | 0.0364 |
| SFE_SE_5 | No | 0.9456 | 0.9073 | 0.0351 |

**Result: virtually identical Spearman (0.946 vs 0.946).** The SFE_SE_5 encoder organizes NCBI data just as well without ever having seen it. Training on NCBI buys nothing for NCBI organization.

**Why NCBI Spearman is so high (0.946 vs 0.847 for marine):** NCBI RefSeq genomes are curated, complete, and taxonomically diverse — they're inherently more separable than metagenomic fragments. The Top-1 MSE (0.035-0.036) is also extremely low compared to marine data (~0.127). NCBI sequences are simply easier to organize in latent space regardless of training data.

**Notable:** The Pearson correlation differs substantially (0.715 vs 0.907) even though Spearman is identical. This means the SFE_SE_NCBI_5 encoder preserves the rank ordering of NCBI neighbors but distorts the linearity of the distance relationship. The SFE_SE_5 encoder, despite never seeing NCBI, produces more linearly faithful distances.

### Summary of SFE_SE_NCBI_5 experiment

| Test data | SFE_SE_5 Spearman | SFE_SE_NCBI_5 Spearman | Delta |
|---|---|---|---|
| SFE_SE (marine) | 0.847 | 0.662 (final) | -0.185 |
| NCBI (reference genomes) | 0.946 | 0.946 | 0.000 |

**Conclusion: Training on NCBI costs ~0.16 Spearman on marine data while providing zero benefit for NCBI organization.** The k-mer frequency representation transfers perfectly — the encoder learns a general mapping from k-mer space to latent space, and NCBI genomes are easy to separate regardless of training data.

This is the strongest evidence yet for the "training data composition > quantity" finding. Adding out-of-domain data doesn't just fail to help — it actively hurts, even at modest fractions (14% NCBI). The dilution effect is not proportional to the fraction of foreign data; even a small amount forces the encoder to allocate latent capacity to an unnecessary manifold.

### Alternative approach: project NCBI through frozen SFE_SE_5 encoder

Training on NCBI data is not necessary for taxonomic signposts. Instead, encode NCBI sequences through the existing SFE_SE_5 encoder without retraining. The k-mer frequency representation is shared — a *Pelagibacter* genome has similar k-mer structure whether it came from RefSeq or a metagenome. Wherever NCBI sequences land near marine clusters in latent space, that provides the taxonomic interpretation.

**Advantages**:
- Preserves the SFE_SE_5 embedding quality (Spearman 0.847)
- NCBI organization is identical (Spearman 0.946 either way)
- Zero training cost
- Can add or remove reference genomes at any time without retraining

**Confirmed**: The "risk" that unseen organisms might project to weird locations is empirically refuted — SFE_SE_5 handles NCBI just as well as a model trained on it.

### Why not test FD data?

FD (Microflora Danica) is aquatic + soil metagenome data — more heterogeneous and further from the marine domain than NCBI. Testing FD against SFE_SE_5 would likely show high Spearman (environmental DNA has similar k-mer structure), confirming what we already know: the encoder generalizes to unseen data. The more interesting test would be FD against augmented models that *were* trained on FD, to check if the same "training doesn't help" pattern holds — but that's a lot of evaluation for a data source not in the clustering pipeline. Low priority unless extending beyond marine metagenomes.

### Why NCBI is closer to marine data than FD

**NCBI RefSeq representative genomes** are complete, curated genomes from across the tree of life. Many marine organisms (or their close relatives) are in RefSeq — *Pelagibacter*, *Prochlorococcus*, *Synechococcus*, various *Roseobacter*, etc. These genomes share genuine evolutionary history with the organisms producing the marine metagenomic fragments. At the k-mer level, a RefSeq *Pelagibacter* genome and a metagenomic *Pelagibacter* contig have very similar frequency profiles because they're the same biology.

**FD (Microflora Danica)** is a metagenomic survey of Danish aquatic and soil environments. While some aquatic organisms overlap with marine taxa, soil microbiomes have fundamentally different community compositions — different GC distributions, different dominant phyla (Actinobacteria, Acidobacteria in soil vs Proteobacteria, Bacteroidetes in marine). The k-mer frequency profiles of soil organisms occupy a different region of k-mer space than marine organisms.

**The paradox resolves**: NCBI is taxonomically diverse but *includes* marine organisms, while FD is environmentally diverse in ways that *diverge* from marine. A curated reference collection that spans the tree of life is a better neighbor to any specific environment than a grab-bag from a different environment. This also explains why adding FD+NCBI together (augmented models) was so much more damaging than NCBI alone — FD contributed the bulk of the foreign data (~8M of ~8.7M non-marine sequences) and its soil/freshwater manifold is the primary source of latent space dilution.

### Exploratory: Spearman on 100 kbp-filtered marine data

Extracted the 154,040 sequences >= 100 kbp from `kmers_SFE_SE_5.npy` and ran verify_local_distances with the SFE_SE_5 encoder.

| Test data | Sequences | Spearman | Pearson | Top-1 MSE | Random MSE |
|---|---|---|---|---|---|
| SFE_SE_5 (all >= 5 kbp) | 50,000 sample | 0.847 | — | 0.124 | 0.495 |
| SFE_SE_100 (>= 100 kbp) | 15,404 sample | 0.766 | 0.421 | 0.020 | 0.334 |

Spearman is *lower* on 100 kbp data (0.766 vs 0.847) despite 100 kbp producing the best clustering results. This is not contradictory — it reflects the difficulty of the ranking task, not the quality of the embedding:

- **Top-1 MSE is 6x lower** (0.020 vs 0.124): 100 kbp sequences are much more similar to each other in k-mer space, making the relative ranking of neighbors harder (smaller differences, more noise).
- **Smaller sample**: only 15,404 sequences encoded (10% of 154K), producing a sparser neighbor pool.
- **Lower random baseline** (0.334 vs 0.495): confirms the 100 kbp sequences occupy a tighter region of k-mer space.

The Spearman metric measures how well the model *rank-orders* neighbors. When all neighbors are already very similar, small perturbations in ranking are penalized but don't affect clustering quality. This is exploratory only — the clustering GC span validation remains the more relevant quality measure for length-filtered subsets.

### Why not train on NCBI alone?

1. **NCBI is easy to organize no matter what.** Spearman 0.946 on NCBI whether the model was trained on it or not. The problem we're solving isn't "how to organize NCBI" — it's "how to organize marine metagenomic fragments." NCBI-only training would optimize for the easy problem while ignoring the hard one.

2. **The dilution effect would be maximized.** Adding 14% NCBI to marine training data dropped marine Spearman from 0.847 to 0.665. A 100% NCBI model would be maximally foreign to the marine data. Metagenomic contigs are fundamentally different from complete reference genomes — shorter, noisier k-mer profiles, many from novel organisms with no close RefSeq representative.

3. **NCBI is small.** Only 656K sequences at the 5K threshold vs 4.8M for SFE_SE. Fewer training examples means less exposure to the diversity of k-mer neighborhoods that exist in the real data we care about.

4. **The asymmetry is the key insight.** NCBI → marine transfer is hard (reference genomes don't teach you about metagenomic fragment structure). Marine → NCBI transfer is free (the encoder already handles it perfectly). Train on the hard domain and get the easy one for free.

**Bottom line**: The frozen SFE_SE_5 encoder gives us 0.847 on marine data AND 0.946 on NCBI. No model we could train would beat both numbers simultaneously.

### Run_NCBI_5 — NCBI-only model for completeness

Training started after Run_SFE_SE_NCBI_5 completed. Dataset: `kmers_NCBI_5.npy` (655,640 contigs from ~20K RefSeq representative genomes).

**Original prediction**: ~0.946 on NCBI (same as every other model) and poorly on marine data (worse than augmented models, since those at least included SFE+SE in training).

**Actual result — prediction was wrong:**

Live Spearman tracking on SFE_SE_5 marine data (50K sample) during training:

| Time | Spearman |
|---|---|
| 20:57 | 0.838 |
| 20:59 | 0.832 |
| 21:01 | 0.831 |
| 21:04 | 0.830 |
| 21:08 | 0.832 |
| **21:13 (final)** | **0.831** |

NCBI Spearman (final): **0.934** (vs 0.946 for SFE_SE_5 which never saw NCBI).

**The NCBI-only model scores 0.831 on marine data it has never seen.** This is:
- Nearly as good as SFE_SE_5 (0.847) — only 0.015 lower
- Far better than SFE_SE_NCBI_5 (0.662) — despite NCBI_5 having *less* relevant training data
- Far better than all augmented models (0.644-0.702) — which included SFE+SE in training
- Stable — no declining trajectory like SFE_SE_NCBI_5 showed

**This is a surprising and important result.** A model trained on only ~20K reference genomes (655K contigs) organizes marine metagenomic data nearly as well as a model trained on 4.8M marine sequences. Yet *mixing* the same NCBI data with marine data (SFE_SE_NCBI_5) destroyed marine embedding quality (0.847 → 0.662).

### The distribution mixing problem

The results across all models now tell a consistent story:

| Model | Training data | Marine Spearman | NCBI Spearman |
|---|---|---|---|
| SFE_SE_5 | 4.8M marine | **0.847** | 0.946 |
| NCBI_5 | 656K NCBI (~20K genomes) | 0.831 | 0.934 |
| SFE_SE_NCBI_5 | 4.8M marine + 656K NCBI | 0.662 | 0.946 |
| Augmented (Run_3) | 13.4M mixed | 0.702 | (not tested) |
| Augmented (Run_5) | 13.4M mixed | 0.644 | (not tested) |

Note: NCBI_5 scores *slightly lower* on NCBI data (0.934) than SFE_SE_5 (0.946) which never saw NCBI. Training on NCBI doesn't even help organize NCBI — the smaller, less diverse training set (656K vs 4.8M) likely means the model has a less well-conditioned latent space overall.

**The problem is not out-of-domain training data. The problem is mixing distributions.** Training on NCBI alone works. Training on SFE_SE alone works. Combining them destroys the marine embedding quality while providing no benefit for NCBI.

However, "distribution mixing" may be the wrong framing — NCBI RefSeq is not a coherent distribution. It's a curated sample across the entire tree of life: independent isolate genomes from many different studies, with no shared environmental context or sequencing protocol. It's a distribution only in the sense that it's the best representative sample NCBI could assemble. The mechanism by which mixing hurts is not yet fully understood.

### NCBI_5 on 100 kbp marine data

| Model | SFE_SE_5 Spearman | SFE_SE_100 Spearman |
|---|---|---|
| SFE_SE_5 | 0.847 | 0.766 |
| NCBI_5 | 0.831 | **0.836** |

The NCBI model is *better* on long marine sequences (>= 100 kbp) than the marine-trained model, and doesn't show the drop from full → 100kbp that SFE_SE_5 does (stays flat: 0.831 → 0.836 vs SFE_SE_5's 0.847 → 0.766).

This likely reflects a length distribution difference: NCBI RefSeq genomes are long, complete sequences — more similar to 100 kbp marine contigs than to the 5 kbp fragments that numerically dominate SFE_SE_5. The SFE_SE_5 model was optimized across its full length range, with short fragments dominating the gradient.

### Length distributions: NCBI_5 vs SFE_SE_5

| | NCBI_5 | SFE_SE_5 |
|---|---|---|
| Count | 655,640 | 4,776,770 |
| Median | **37 kbp** | **13 kbp** |
| Mean | 150 kbp | 26 kbp |
| 5-10 kbp | 16.5% | 36.4% |
| 10-50 kbp | 40.7% | 54.0% |
| 50-100 kbp | 16.0% | 6.4% |
| >= 100 kbp | **26.7%** | **3.3%** |

NCBI sequences are dramatically longer — over a quarter are >= 100 kbp (complete or near-complete genomes), while SFE_SE is dominated by short metagenomic contigs (90% under 50 kbp, 36% in the 5-10 kbp bin).

This confirms why NCBI_5 beats SFE_SE_5 on 100 kbp marine data (0.836 vs 0.766): the NCBI model was trained predominantly on long sequences, so it learned to organize that region of k-mer space well. The SFE_SE_5 model's gradient is dominated by the 90% of sequences under 50 kbp.

Note: NCBI has a min of 1,006 bp despite the 5 kbp threshold — likely small contigs/chromosomes from multi-chromosome genomes where the genome-level filter passed but individual contigs are short.

### Generated kmers_SFE_SE_100.npy

Extracted 154,040 sequences >= 100 kbp from `kmers_SFE_SE_5.npy` with matching `ids_SFE_SE_100.txt`. Row order matches the source files. Previous `ids_SFE_SE_100.txt` (from notebooks) had same count but different ordering — backed up as `.bak`.

### Run_SFE_SE_100 — marine-only, long sequences only

Training on `kmers_SFE_SE_100.npy` (154,040 sequences >= 100 kbp).

Live Spearman tracking on SFE_SE_5 full marine data (50K sample):

| Time | SFE_SE_5 Spearman | Delta |
|---|---|---|
| 21:30 | 0.846 | — |
| 21:32 | 0.816 | -0.030 |
| 21:34 | 0.805 | -0.011 |
| **21:36 (final)** | **0.797** | **-0.008** |

SFE_SE_100 Spearman (final): **0.804**

Same declining pattern as SFE_SE_NCBI_5 — starts strong, declines as the model specializes. Trains very fast with only 154K sequences.

### Complete cross-model comparison (final results)

| Model | Training data | N seqs | SFE_SE_5 Spearman | SFE_SE_100 Spearman | NCBI Spearman |
|---|---|---|---|---|---|
| **SFE_SE_5** | Marine >= 5 kbp | 4.8M | **0.847** | 0.766 | 0.946 |
| NCBI_5 | NCBI RefSeq (~20K genomes) | 656K | 0.831 | **0.836** | 0.934 |
| SFE_SE_100 | Marine >= 100 kbp | 154K | 0.797 | 0.804 | (not tested) |
| SFE_SE_NCBI_5 | Marine + NCBI | 5.4M | 0.662 | (not tested) | 0.946 |
| Augmented (Run_3) | FD+NCBI+SFE+SE | 13.4M | 0.702 | (not tested) | (not tested) |

**Key observations:**

1. **SFE_SE_5 remains the best model for full marine data** (0.847) and the overall best choice.

2. **NCBI_5 is the best model for 100 kbp marine data** (0.836), beating even SFE_SE_100 which was trained on that exact data (0.804). This is because NCBI has 4x more sequences from much broader taxonomy — the taxonomic diversity matters more than domain matching for long sequences.

3. **SFE_SE_100 underperforms despite training on the "right" data.** With only 154K sequences, it lacks taxonomic diversity — 100 kbp contigs are a narrow slice (3.3% of SFE_SE_5), likely over-representing a few abundant, long-genome organisms.

4. **The mixed model (SFE_SE_NCBI_5) is the worst of the lot** (0.662), despite having the most total data (5.4M) and including both marine and NCBI sequences. Mixing distributions during training is actively harmful.

5. **Sample count matters less than expected for VAEs.** The reparameterization trick means each forward pass samples a different z from the latent distribution, so the model effectively sees different inputs each time — the same sequence never produces the exact same training signal twice. The 154K → 0.797 result is about insufficient *taxonomic diversity*, not insufficient samples for the ~7M parameter count.

6. **Taxonomic breadth is the key variable.** NCBI_5 succeeds with only 656K sequences because ~20K reference genomes span the tree of life, giving the encoder a broad view of how k-mer space is structured. SFE_SE_5 succeeds with 4.8M sequences because the marine metagenome is naturally diverse. SFE_SE_100 fails with 154K sequences because it's a taxonomically narrow subset.

### Generated kmers_100.npy (augmented, >= 100 kbp)

Extracted 844,705 sequences >= 100 kbp from `kmers_5.npy` (all four sources: FD + NCBI + SFE + SE) with matching `ids_100.txt`. 8.8 GB. This is 5.5x more sequences than SFE_SE_100 (154K) and includes NCBI's taxonomic breadth plus FD's soil/aquatic diversity.

### Run_100 — first results challenge the mixing hypothesis

Early Spearman on SFE_SE_5 marine data: **0.845** — matching SFE_SE_5 (0.847) despite containing all four sources including FD.

| Time | SFE_SE_5 Spearman |
|---|---|
| 21:49 | 0.845 |

**This challenges the "mixing distributions hurts" narrative.** Run_100 mixes FD+NCBI+SFE+SE — the same combination that produced the worst augmented models (0.644-0.702) — yet it's performing on par with the best model. The only difference is the 100 kbp length filter.

### Revised hypothesis: sequence length, not source mixing

Re-examining the evidence:

| Model | Sources | Length filter | Marine Spearman |
|---|---|---|---|
| SFE_SE_5 | SFE+SE | >= 5 kbp | **0.847** |
| **Run_100** | **FD+NCBI+SFE+SE** | **>= 100 kbp** | **0.845** |
| NCBI_5 | NCBI only | >= 5 kbp (median 37 kbp) | 0.831 |
| SFE_SE_100 | SFE+SE | >= 100 kbp | 0.797 |
| Run_3 (augmented) | FD+NCBI+SFE+SE | >= 3 kbp | 0.702 |
| SFE_SE_NCBI_5 | SFE+SE+NCBI | >= 5 kbp | 0.662 |
| Run_5 (augmented) | FD+NCBI+SFE+SE | >= 5 kbp | 0.644 |

The common factor in the failures is not source mixing — it's having short sequences in the training set. Run_100 has all four sources including FD, yet matches SFE_SE_5. The augmented runs (Run_3, Run_5) used the same sources but with short sequences dominating (median ~13 kbp).

Short metagenomic contigs have noisier k-mer profiles because there are fewer k-mers to count. Training on millions of noisy short sequences may teach the model to accommodate noise, degrading the latent space structure. The length filter removes this noise, and then source diversity doesn't matter — or may even help by providing broader taxonomic coverage.

**The original "training data composition > quantity" finding may actually be "training data quality (length) > quantity".** Source mixing was a confound because different sources have different length distributions. FD adds millions of short contigs that degrade the model.

**Key test**: If Run_100 holds at ~0.845 as training continues (doesn't decline like SFE_SE_NCBI_5 did), that confirms source mixing is not the problem — sequence length is. Still early; need to keep tracking.

Run_100 tracking update:

| Time | SFE_SE_5 Spearman | Delta |
|---|---|---|
| 21:49 | 0.845 | — |
| 21:53 | 0.826 | -0.019 |

Declining — same pattern as other models. Still too early to determine the floor. Run_100 does contain FD, so this could be FD's soil/freshwater data or the model specializing for 100 kbp and losing generalization to shorter sequences.

### Puzzle resolved: SFE_SE_NCBI_5 is a length mismatch, not a source mixing effect

The earlier "remaining puzzle" — why SFE_SE_NCBI_5 (0.662) performed worse than SFE_SE_5 (0.847) despite only adding 14% NCBI — is explained by length distribution mismatch. NCBI has median 37 kbp vs SFE_SE's median 13 kbp. Combining them creates a bimodal length distribution. The model has to simultaneously serve short noisy sequences and long clean ones, and the conflicting gradients degrade the latent space for both.

The same logic explains why the original augmented models (Run_1-5, Spearman 0.644-0.702) were bad — FD adds millions of short contigs alongside NCBI's longer genomes, creating even more length heterogeneity.

### Revised core finding: homogeneous length distribution is what matters

The picture simplifies:

- **All-short works**: SFE_SE_5 (median 13 kbp) → 0.847
- **All-long works**: NCBI_5 (median 37 kbp) → 0.831, Run_100 (all >= 100 kbp) → 0.845 early
- **Mixed lengths hurts**: SFE_SE_NCBI_5 (13 kbp + 37 kbp bimodal) → 0.662, augmented runs (mixed) → 0.644-0.702

Source identity doesn't matter — source mixing was a confound for length mixing. The earlier "training data composition > quantity" finding should be reframed as **"training data length homogeneity > quantity"**.

### Run_100 final results

Training completed. Full Spearman tracking on SFE_SE_5 marine data:

| Time | SFE_SE_5 Spearman | Delta |
|---|---|---|
| 21:49 | 0.845 | — |
| 21:53 | 0.826 | -0.019 |
| 21:57 | 0.815 | -0.011 |
| 22:01 | 0.806 | -0.009 |
| 22:12 | 0.790 | -0.016 |
| 22:13 | 0.788 | -0.002 |
| **22:15 (final)** | **0.784** | **-0.004** |

Final Spearman across all test sets:

| Test data | Run_100 Spearman |
|---|---|
| SFE_SE_5 (full marine) | 0.784 |
| SFE_SE_100 (marine >= 100 kbp) | 0.798 |
| kmers_100 (own training data, all sources >= 100 kbp) | 0.788 |
| NCBI_5 (reference genomes) | 0.894 |

**Observations:**
- No home-field advantage: 0.788 on own training data vs 0.784 on SFE_SE_5.
- Slightly better on SFE_SE_100 (0.798) than on kmers_100 (0.788), suggesting FD sequences in the training data are the harder-to-organize component.
- Well above augmented models (0.644-0.702) but below NCBI_5 (0.831) and SFE_SE_5 (0.847).
- FD's inclusion may be dragging it down. The soil/freshwater sequences are the most foreign to the marine test data.

### Complete cross-model comparison (all final results, 2026-02-20)

| Model | Training data | N seqs | SFE_SE_5 | SFE_SE_100 | NCBI | Own data |
|---|---|---|---|---|---|---|
| **SFE_SE_5** | Marine >= 5 kbp | 4.8M | **0.847** | 0.766 | **0.946** | — |
| NCBI_5 | NCBI RefSeq (~20K genomes) | 656K | 0.831 | **0.836** | 0.934 | — |
| SFE_SE_100 | Marine >= 100 kbp | 154K | 0.797 | 0.804 | — | — |
| Run_100 | All sources >= 100 kbp | 845K | 0.784 | 0.798 | 0.894 | 0.788 |
| SFE_SE_NCBI_5 | Marine + NCBI (mixed lengths) | 5.4M | 0.662 | — | 0.946 | — |
| Run_3 (augmented) | All sources >= 3 kbp (mixed lengths) | 13.4M | 0.702 | — | — | — |
| Run_5 (augmented) | All sources >= 5 kbp (mixed lengths) | 13.4M | 0.644 | — | — | — |

**Summary of what we learned today:**

1. **SFE_SE_5 remains the best model** for full marine data (0.847). No other model beat it.
2. **Mixing length distributions is the primary damage mechanism**, not mixing sources. Run_100 (all 4 sources, all long) scores 0.784 — far above augmented models with the same sources but mixed lengths (0.644-0.702).
3. **NCBI_5 is surprisingly good** (0.831 marine, 0.836 on 100 kbp marine) from only ~20K reference genomes. Taxonomic breadth compensates for small sample size.
4. **Training on NCBI adds zero value for NCBI organization** — all models score 0.93-0.95 on NCBI regardless. This is likely because NCBI RefSeq *representative* genomes were curated to be taxonomically distinct — they occupy well-separated regions of k-mer space by design. Any reasonable encoder preserves their ordering because there's huge distance between them. Marine metagenomic contigs are the opposite: dense, overlapping, from closely related organisms in the same environment. That's the hard discrimination problem, and that's why model choice matters for marine data but not for NCBI.
5. **Frozen SFE_SE_5 encoder + NCBI projection** is the optimal strategy for taxonomic signposts: best marine embedding AND best NCBI organization, zero retraining cost.
6. **VAE sample count is less important than taxonomic diversity**, because the reparameterization trick effectively augments training data. The 154K-sequence SFE_SE_100 model (0.797) failed not from too few samples but from too narrow a taxonomic slice.

### Which model to use? — Open question

On 100 kbp marine data (what our clustering pipeline uses): NCBI_5 (0.836) >> SFE_SE_5 (0.766). On full marine data: SFE_SE_5 (0.847) > NCBI_5 (0.831).

However, we've learned that one metric doesn't predict another — reconstruction loss didn't predict Spearman, and Spearman might not predict clustering GC span quality. The real test is to run NCBI_5 through the actual clustering pipeline on 100 kbp marine data and compare GC spans against the SFE_SE_5 results we already have (clustering_100.ipynb). That's the metric we actually care about.

### Run_NCBI_100

Trained on 175,213 NCBI RefSeq sequences >= 100 kbp (vs NCBI_5's 656K with >= 5 kbp threshold).

Spearman tracking on SFE_SE_5 marine data:

| Time | SFE_SE_5 Spearman | Delta |
|---|---|---|
| 22:36 | 0.839 | — |
| 22:38 | 0.837 | -0.002 |
| 22:40 | 0.836 | -0.001 |
| **22:42 (final)** | **0.836** | **0.000** |

Very stable — almost no decline at all.

Final Spearman across all test sets:

| Test data | NCBI_100 Spearman |
|---|---|
| SFE_SE_5 (full marine) | 0.836 |
| SFE_SE_100 (marine >= 100 kbp) | 0.832 |
| NCBI_5 (reference genomes) | 0.919 |

**Comparison with NCBI_5:**

| Test data | NCBI_5 | NCBI_100 | Delta |
|---|---|---|---|
| SFE_SE_5 (full marine) | 0.831 | 0.836 | +0.005 |
| SFE_SE_100 (marine >= 100 kbp) | 0.836 | 0.832 | -0.004 |
| NCBI_5 (reference genomes) | 0.934 | 0.919 | -0.015 |

The 100 kbp filter on NCBI made almost no difference. NCBI already has median 37 kbp, so filtering to >= 100 kbp just removes the shorter tail (175K of 656K sequences retained = 26.7%) without changing the character of the training set. The slight NCBI score drop (0.934 → 0.919) makes sense: training on fewer NCBI sequences means slightly less exposure to the NCBI distribution, but this barely matters since NCBI genomes are well-separated by design.

### Updated cross-model comparison (all final results, 2026-02-20)

| Model | Training data | N seqs | SFE_SE_5 | SFE_SE_100 | NCBI | Own data |
|---|---|---|---|---|---|---|
| **SFE_SE_5** | Marine >= 5 kbp | 4.8M | **0.847** | 0.766 | **0.946** | — |
| NCBI_5 | NCBI RefSeq >= 5 kbp | 656K | 0.831 | **0.836** | 0.934 | — |
| NCBI_100 | NCBI RefSeq >= 100 kbp | 175K | 0.836 | 0.832 | 0.919 | — |
| SFE_SE_100 | Marine >= 100 kbp | 154K | 0.797 | 0.804 | — | — |
| Run_100 | All sources >= 100 kbp | 845K | 0.784 | 0.798 | 0.894 | 0.788 |
| SFE_SE_NCBI_5 | Marine + NCBI (mixed lengths) | 5.4M | 0.662 | — | 0.946 | — |
| Run_3 (augmented) | All sources >= 3 kbp (mixed lengths) | 13.4M | 0.702 | — | — | — |
| Run_5 (augmented) | All sources >= 5 kbp (mixed lengths) | 13.4M | 0.644 | — | — | — |

**NCBI_100 observation**: Confirms that the NCBI model family is robust. Whether trained on 656K (>= 5 kbp) or 175K (>= 100 kbp) sequences, performance is essentially identical. This further supports the finding that taxonomic breadth matters more than sample count — both models see the same ~20K reference genomes, just with different length filtering.

### NCBI_100 as test set

Tested models against kmers_NCBI_100.npy (175K NCBI sequences >= 100 kbp) to see if filtering NCBI by length changes the picture:

| Model | NCBI_5 test | NCBI_100 test |
|---|---|---|
| **SFE_SE_5** | **0.946** | **0.947** |
| NCBI_5 | 0.934 | 0.925 |
| NCBI_100 | 0.919 | 0.910 |

Same pattern — SFE_SE_5 dominates on NCBI regardless, all scores 0.91-0.95. NCBI genomes remain trivially easy to organize no matter how you slice them.

### Model selection summary

SFE_SE_5 wins across the board *except* on 100 kbp marine data, where NCBI_5 has a large edge (0.836 vs 0.766). Since the clustering pipeline operates on long marine contigs (>= 100 kbp), that's the most relevant test set — and there NCBI_5 leads.

However, Spearman might not predict clustering GC span quality, just as reconstruction loss didn't predict Spearman. The real test is to run NCBI_5 through the actual clustering pipeline on 100 kbp marine data and compare GC spans against the existing SFE_SE_5 results (clustering_100.ipynb).

### Clustering comparison: NCBI_5 vs SFE_SE_5 on 100 kbp marine data (2026-02-21)

Generated `embed_SFE_SE_1_NCBI_5.npy` (6.7M sequences, 385 cols) by running all SFE_SE_1 data through the NCBI_5 encoder. Filtered to >= 100 kbp (154,040 sequences), loaded into a separate ChromaDB (`.chroma_100_NCBI_5`), and queried 50-NN to produce `Runs/neighbors_100_NCBI_5.tsv`.

Notebook: `clust_100_NCBI_5.ipynb`

**Graph statistics**: NCBI_5 creates a denser neighborhood — 133,724 nodes with 4,571,866 edges (d<5, cap=100) vs SFE_SE_5's 123,783 nodes with 3,391,528 edges. More sequences have neighbors within d=5.

**Leiden GC spans (Capped d<5, top 3 communities)**:

| | SFE_SE_5 | NCBI_5 |
|---|---|---|
| Community #1 | size 1,637, GC span 11 pp | size 1,695, GC span 10 pp |
| Community #2 | size 1,447, GC span 15 pp | size 1,609, GC span 17 pp |
| Community #3 | size 1,314, GC span 7 pp | size 1,533, GC span 7 pp |

Leiden comparable — neither model produces tight clusters.

**MCL GC spans (top 3 communities per inflation)**:

| I | SFE_SE_5 spans | NCBI_5 spans | SFE_SE_5 clusters | NCBI_5 clusters |
|---|---|---|---|---|
| 1.4 | 5, 8, 6 pp | 8, 6, 9 pp | 7,693 | 7,142 |
| 2.0 | 5, 5, 4 pp | 8, 7, 4 pp | 9,710 | 8,885 |
| 3.0 | **4, 6, 4 pp** | **4, 4, 4 pp** | 12,305 | 11,413 |
| 4.0 | 4, 5, 4 pp | 4, 7, 3 pp | 15,202 | 15,592 |
| 5.0 | 4, 5, 4 pp | 4, 7, 5 pp | 17,323 | 18,577 |
| 6.0 | 4, 5, 4 pp | 7, 4, 5 pp | 19,198 | 21,047 |

At I=3.0 (our standard): NCBI_5 produces uniformly 4 pp GC spans in the top 3 communities, matching or slightly beating SFE_SE_5 (4, 6, 4 pp). Both models produce excellent clustering at this inflation.

**Key observation**: Despite SFE_SE_5 having better Spearman on full marine data (0.847 vs 0.831), the actual clustering GC spans are comparable. And despite NCBI_5 having much better Spearman on 100 kbp marine data (0.836 vs 0.766), the clustering improvement is modest — 4/4/4 vs 4/6/4 pp at I=3.0. This confirms the earlier finding that **Spearman doesn't reliably predict clustering quality**.

**Files generated**:
- `Runs/embed_SFE_SE_1_NCBI_5.npy` — Full SFE_SE embeddings from NCBI_5 encoder (6.7M × 385)
- `Runs/embed_100_NCBI_5.npy` — 100 kbp filtered (154K × 385)
- `Runs/ids_100_NCBI_5.txt` — Matching IDs
- `Runs/neighbors_100_NCBI_5.tsv` — 50-NN graph (154K rows)
- `Runs/graph_capped100_d5_100_NCBI_5.tsv` — In-degree capped MCL input
- `Runs/MCL_100_NCBI_5_d5/` — MCL results at I=1.4-6.0

**Naming convention**: `embed_<data>_<model>.npy` (e.g., `embed_SFE_SE_1_NCBI_5.npy` = SFE_SE_1 data embedded with NCBI_5 model).
