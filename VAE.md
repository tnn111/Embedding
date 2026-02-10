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

### Critical Issues

**1. Inference scripts use stochastic `z` instead of deterministic `z_mean`**

Both `embedding` (line 118) and `create_and_load_db` (line 164) use the sampled `z` output from the encoder rather than `z_mean`. The stochastic `z` includes reparameterization noise, meaning the same input produces different embeddings each run. Standard VAE inference practice is to use `z_mean`. Notably, `verify_local_distances.py` correctly uses `z_mean` — so the validation measured quality under a different regime than the deployed pipeline.

**2. `verify_local_distances.py` has stale column indices**

Still uses `COL_START = 8193`, `COL_END = 10965` (old 7-mer format). Current data uses `COL_START = 1`, `COL_END = 2773`. Script would produce wrong results on current data.

**3. `convert_txt_to_npy` is outdated**

References `NUM_COLUMNS = 2762` and a format with a "GC" column that no longer exists.

### Code Quality Issues

**4. Duplicated custom layers across 4 files**

`ClipLayer`, `Sampling`, and `clr_transform` are copy-pasted into VAE.py, `embedding`, `create_and_load_db`, and `verify_local_distances.py`. Divergence has already caused bug #2.

**5. Stale model symlink**

Root `vae_encoder_best.keras` symlink points to `Models/Multi_005_384/` (Dec 3, 4.8M dataset). Best model (13.4M dataset, dramatically better metrics) lives in `Data/`. Inference scripts resolve to the old model.

**6. `main.py` is a placeholder**

Just prints "Hello from clustering!" — leftover from `uv init`.

### Architecture & Training Notes

**7. CLR applied across mixed compositions**

`calculate_kmer_frequencies` normalizes each k-size group independently (each sums to 1.0), then VAE.py applies CLR to all 2,772 features together, mixing 6 separate compositions. CLR is designed for a single compositional vector. Per-group CLR would be more theoretically sound. Current approach works well empirically — the joint CLR may create useful cross-scale interactions.

**8. No dropout — decided not to add**

Architecture relies on BatchNorm + KL regularization. Dropout was considered but deemed unnecessary: train/val gap is small (98 vs 101), KL term already prevents memorization, and data-to-parameter ratio is healthy (13.4M samples / 7M params). Adding dropout would likely hurt reconstruction quality without meaningful generalization benefit. Worth revisiting only if length sweep reveals overfitting on smaller datasets.

**9. Training history not merged across runs**

`vae_history.pkl` is overwritten on each resume, losing the full training curve.

**10. Small CLR pseudocount (1e-6)**

For 5,000 bp sequences, many 6-mers have zero counts, producing CLR values near -13.8. A larger pseudocount (e.g., 1e-3 or 1/N) would reduce dynamic range extremes.

### Minor Issues

- pyproject.toml project name is "clustering"
- README.md is empty
- `embedding` script hardcodes encoder path (no CLI option unlike `create_and_load_db`)
- Models scattered across `Models/`, `Data/`, and project root
- `NumpyBatchDataset` drops last incomplete batch (negligible)

### Prioritized Suggestions

1. Fix inference to use `z_mean` (highest impact, one-line fix each)
2. Update `verify_local_distances.py` column indices
3. Update model symlink to point to best current model
4. Extract shared code (`ClipLayer`, `Sampling`, `clr_transform`) to a module
5. Remove/update stale files (`convert_txt_to_npy`, `main.py`)
6. ~~Consider per-group CLR as an experiment~~ — **Done** (see below)

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

### Setup
- 5 datasets filtered at 1,000 / 2,000 / 3,000 / 4,000 / 5,000 bp minimum
- Sources: FD (Microflora Danica), NCBI RefSeq, SE (Baltic), SFE (San Francisco Estuary)
- FD contigs already filtered at 3 kbp before ENA submission, so 1k and 2k thresholds only add shorter contigs from other sources
- All runs: 1,000 epochs, beta=0.05, per-group CLR with Jeffreys prior
- verify_local_distances.py run with 50k sample size on each run's own training data

### Training results

| Min Length | Val Loss | MSE | 6-mer | 5-mer | 4-mer | 3-mer | 2-mer | 1-mer | KL | Time | Peak RAM |
|-----------|---------|------|-------|-------|-------|-------|-------|-------|-----|------|----------|
| 1,000 bp | 252.2 | 0.056 | 0.0727 | 0.0082 | 0.0034 | 0.0019 | 0.0009 | 0.0002 | 490 | 10h56m | 326 GB |
| 2,000 bp | 239.3 | 0.067 | 0.0867 | 0.0099 | 0.0047 | 0.0027 | 0.0011 | 0.0001 | 509 | 8h19m | 317 GB |
| 3,000 bp | 207.3 | 0.064 | 0.0834 | 0.0085 | 0.0043 | 0.0029 | 0.0014 | 0.0001 | 520 | 8h24m | 306 GB |
| 4,000 bp | 174.6 | 0.051 | 0.0662 | 0.0055 | 0.0026 | 0.0013 | 0.0005 | 0.0001 | 502 | 7h16m | 276 GB |
| 5,000 bp | 162.4 | 0.051 | 0.0656 | 0.0071 | 0.0024 | 0.0014 | 0.0006 | 0.0001 | 466 | 6h42m | 252 GB |

### Local distance verification (Spearman correlation)

| Min Length | Spearman r | Pearson r | Top 1 MSE | Top 50 MSE | Random MSE | NN/Random ratio |
|-----------|-----------|-----------|-----------|-----------|-----------|----------------|
| 1,000 bp | **0.852** | **0.778** | 0.121 | 0.165 | 0.555 | 4.6x |
| 2,000 bp | 0.742 | 0.487 | 0.129 | 0.168 | 0.521 | 4.0x |
| 3,000 bp | 0.714 | 0.611 | 0.190 | 0.244 | 0.550 | 2.9x |
| 4,000 bp | 0.650 | 0.363 | 0.140 | 0.259 | 0.516 | 3.7x |
| 5,000 bp | 0.731 | 0.463 | 0.116 | 0.170 | 0.492 | 4.2x |

### Observations

- **Run 1 (1,000 bp) has the best Spearman correlation** (0.852) by a wide margin, confirming the Jeffreys prior solved the short-contig problem
- Results are not monotonic — Run 4 (4,000 bp) has the worst Spearman (0.650), suggesting interactions between dataset size, diversity, and contig quality
- KL is higher for shorter-contig runs (490-520 vs 466), indicating the model uses more latent capacity for noisier inputs
- Val loss decreases with longer minimum length (252 → 162), but this partly reflects fewer sequences and less diversity to model
- The previous model (same preprocessing, different data mix) achieved Spearman 0.93 on its own training data — all sweep runs are lower, possibly due to different source compositions

### Cross-comparison: all models tested on same 5,000 bp data (kmers_5.npy)

Testing each model's own data is not a fair comparison since the data distributions differ. Here all 5 models are tested on the same 5,000 bp dataset (50k sample):

| Trained on | Spearman r | Pearson r | Top 1 MSE | Top 50 MSE |
|-----------|-----------|-----------|-----------|-----------|
| 1,000 bp | 0.694 | 0.408 | 0.106 | 0.165 |
| 2,000 bp | 0.712 | 0.441 | 0.107 | 0.163 |
| 3,000 bp | 0.717 | 0.432 | 0.108 | 0.170 |
| ~~4,000 bp~~ | ~~0.580~~ | ~~0.296~~ | ~~0.126~~ | ~~0.226~~ |
| 4,000 bp (Run 4') | 0.727 | 0.453 | 0.109 | 0.167 |
| **5,000 bp** | **0.731** | **0.463** | **0.116** | **0.170** |

Random baseline: MSE = 0.492 (same across all, since test data is identical)

**Interpretation:**
- Run 5 (5,000 bp) is the best on 5,000 bp test data, as expected — trained on matching distribution
- With Run 4' corrected, Spearman increases monotonically with threshold: 0.694 → 0.712 → 0.717 → 0.727 → 0.731
- The original Run 4 outlier at 0.580 was a ReduceLROnPlateau scheduling artifact (see below)
- Including shorter contigs in training barely hurts performance on long contigs (Spearman drop from 0.731 to ~0.69 is modest)
- Training on shorter contigs doesn't significantly degrade the model's ability to embed long contigs

### Run 4 analysis: learning rate scheduling artifact

Run 4 (4,000 bp) is a clear outlier at Spearman 0.580 on the cross-comparison. Deep analysis of the training logs reveals this is a **learning rate scheduling artifact**, not a convergence or threshold effect.

**ReduceLROnPlateau configuration:** `patience=20, factor=0.5, min_lr=1e-6`

**LR reduction schedule comparison:**

| Run | Min bp | 1st LR (→5e-5) | Last LR (→1e-6) | Epochs at min LR |
|-----|--------|----------------|-----------------|------------------|
| Run 1 | 1,000 | Epoch 21 | Epoch 339 | 661 |
| Run 2 | 2,000 | Epoch 22 | Epoch 387 | 613 |
| Run 3 | 3,000 | Epoch 22 | Epoch 453 | 547 |
| **Run 4** | **4,000** | **Epoch 566** | **Epoch 854** | **146** |
| Run 5 | 5,000 | Epoch 23 | Epoch 391 | 609 |

Run 4's first LR reduction was 27x later than all other runs. It spent 566 epochs at LR=1e-4 while all others dropped to 5e-5 by epoch ~22. Run 4 only had 146 epochs at minimum LR vs 547-661 for others.

**Why didn't the scheduler trigger?** Run 4's val_loss was on a long, slow descent that kept improving every <20 epochs, never triggering the patience:

```
Epoch  43: val_loss = 185.32
Epoch  91: val_loss = 183.54   (−1.78 over 48 epochs)
Epoch 164: val_loss = 182.36   (−1.18 over 73 epochs)
Epoch 270: val_loss = 181.07   (−1.29 over 106 epochs)
Epoch 360: val_loss = 179.47   (−1.60 over 90 epochs)
Epoch 482: val_loss = 177.26   (−2.21 over 122 epochs)
Epoch 546: val_loss = 176.31   (−0.95 over 64 epochs)
--- 20 epochs with no improvement → 1st LR reduction at epoch 566 ---
Epoch 567: val_loss = 175.84   (immediately improves after LR drop)
```

The model saved improved models all the way to epoch 1000 (val_loss: 193.08 → 174.63).

**Paradox: Run 4 has the best reconstruction but worst retrieval.**

| Run | 6-mer | 5-mer | 4-mer | 3-mer | 2-mer | KL |
|-----|-------|-------|-------|-------|-------|-----|
| Run 1 | 0.0727 | 0.0082 | 0.0034 | 0.0019 | 0.0009 | 490 |
| Run 2 | 0.0867 | 0.0099 | 0.0047 | 0.0027 | 0.0011 | 509 |
| Run 3 | 0.0834 | 0.0085 | 0.0043 | 0.0029 | 0.0014 | 520 |
| **Run 4** | **0.0662** | **0.0055** | **0.0026** | **0.0013** | **0.0005** | 502 |
| Run 5 | 0.0656 | 0.0071 | 0.0024 | 0.0014 | 0.0006 | 466 |

Run 4 has the best or near-best MSE across all k-mer sizes (dominant on 5-mer, 3-mer, 2-mer) but the worst Spearman correlation. Classic reconstruction-representation tradeoff.

**Mechanism:** The other runs quickly reduced LR, allowing latent space local neighborhoods to stabilize early and refine over 550-660 epochs at min LR. Run 4 stayed at high LR for 566 epochs, continuously reorganizing the latent space. The high LR prevented fine-grained local structure from forming. By the time it reached min LR, only 146 epochs remained — not enough to develop well-organized neighborhoods.

**Fix options:**
1. Retrain Run 4 with a fixed LR schedule matching other runs (manual reduction at ~epoch 25)
2. Extend training for 1000+ more epochs at min LR (uncertain benefit — latent structure learned at high LR may be fundamentally different)
3. Accept as a scheduler artifact and note in the paper

**Fix chosen:** Retraining Run 4 as Run_4_prime with starting LR=5e-5 (matches the effective LR the other runs used after their first reduction at epoch ~22). With 1000 epochs and ReduceLROnPlateau still in place, this should produce a comparable training regime to the other runs. Run started 2026-02-08.

**Run_4_prime early results (in progress):**
LR schedule is now behaving normally:

| LR tier | Run 1 | Run 2 | Run 3 | Run 5 | Run 4' |
|---------|-------|-------|-------|-------|--------|
| → 2.5e-5 | 146 | 201 | 225 | 148 | 266 |
| → 1.25e-5 | 170 | 253 | 253 | 194 | 292 |
| → 6.25e-6 | 232 | 276 | 301 | 233 | 365 |
| → 3.125e-6 | 286 | 339 | 352 | 319 | 388 |
| → 1.56e-6 | 311 | 364 | 383 | 345 | 409 |
| (original Run 4: 1st reduction at epoch 566) |||||

Normal LR schedule. Min LR (1e-6) reached at epoch 459, giving 541 epochs of fine-tuning — comparable to the other runs:

| Run | Min LR at | Epochs at min LR |
|-----|-----------|-----------------|
| Run 1 | 339 | 661 |
| Run 5 | 391 | 609 |
| Run 2 | 387 | 613 |
| Run 3 | 453 | 547 |
| **Run 4'** | **459** | **541** |
| Run 4 (original) | 854 | 146 |

**Final Run_4_prime results (1000 epochs):**

| Model | Spearman (5k data) | Spearman (own data) |
|-------|-------------------|-------------------|
| Run 4 (original) | 0.580 | 0.650 |
| Run 1 (1k) | 0.694 | 0.852 |
| Run 2 (2k) | 0.712 | 0.742 |
| SFE_SE_5 | 0.714 | 0.742 |
| Run 3 (3k) | 0.717 | 0.714 |
| **Run 4' (4k)** | **0.727** | **0.786** |
| Run 5 (5k) | 0.731 | 0.731 |

Run 4' lands at Spearman 0.727 on the common 5k test data — between Run 3 (0.717) and Run 5 (0.731), exactly where the 4,000 bp threshold should fall. The original Run 4's 0.580 was purely a ReduceLROnPlateau scheduling artifact. Case closed.

### FD contig length filtering

The Microflora Danica paper (doi:10.1038/s41564-025-02062-z) uses mmlong2 pipeline which filters contigs at 3,000 bp minimum by default. FD contigs obtained from ENA were already pre-filtered at 3 kbp before submission. This means the 1,000 bp and 2,000 bp sweep runs only gain shorter contigs from non-FD sources (aquatic metagenomes, RefSeq). The 3,000 bp threshold is standard for metagenomic binning workflows.

### Conclusions

- The Jeffreys prior solved the short-contig problem — 1,000 bp contigs are now viable
- With Run 4' corrected, Spearman on 5k test data increases monotonically with threshold: 0.694 → 0.712 → 0.717 → 0.727 → 0.731
- The spread is modest (0.694-0.731) — minimum contig length has limited impact on embedding quality
- The original Run 4 outlier was a ReduceLROnPlateau scheduling artifact, confirmed by retraining with starting LR=5e-5
- A 3,000 bp threshold aligns with standard metagenomic practice (mmlong2, FD paper)
- Choice of threshold depends on use case: 3,000 bp for compatibility with existing pipelines, lower thresholds if short contigs are important
- Reconstruction loss alone is insufficient to evaluate embedding quality — Run 4 had the best MSE but worst Spearman

---

## 2026-02-08: Run_SFE_SE — aquatic-only baseline with new preprocessing

### Setup
- **Data:** SFE + SE contigs ≥ 5,000 bp only (4,776,770 samples) — same data as the original 4.8M aquatic model
- **Preprocessing:** Per-group CLR + Jeffreys prior (new pipeline)
- **Training:** 1,000 epochs, same architecture

### Results

| Epoch | Val | MSE | 6-mer | 5-mer | 4-mer | 3-mer | 2-mer | 1-mer | KL |
|-------|-----|-----|-------|-------|-------|-------|-------|-------|-----|
| 50 | 194.7 | 0.058 | 0.0751 | 0.0100 | 0.0022 | 0.0011 | 0.0006 | 0.0001 | 398 |
| 200 | 191.0 | 0.058 | 0.0741 | 0.0097 | 0.0026 | 0.0015 | 0.0006 | 0.0003 | 404 |
| 500 | 185.0 | 0.055 | 0.0718 | 0.0079 | 0.0013 | 0.0006 | 0.0003 | 0.0000 | 403 |
| 1000 | 184.8 | 0.055 | 0.0717 | 0.0079 | 0.0013 | 0.0006 | 0.0003 | 0.0000 | 403 |

### LR schedule
- 1st reduction at epoch 248, min LR at epoch 535 (465 epochs at min LR)
- Later than sweep runs (21-23) but much earlier than Run 4 (566)
- Fewer samples per epoch → slower per-epoch progress → later plateau

### Per-k-mer patterns
- **6-mer worse** (0.0717 vs 0.0656 Run 5) — less diverse training data, 6-mer has the most features to learn
- **4-mer through 1-mer dramatically better** — 4-mer 0.0013 vs 0.0024 (Run 5), nearly half. With only 2 environments, the model specializes on shorter k-mers
- **KL = 403** — lowest of any run (vs 466-520 for sweep runs). Two environments need less latent capacity

### Local distance verification (Spearman)

**On own data (SFE_SE, 50k sample):** Spearman = 0.742, Pearson = 0.372

**Cross-comparison on common 5k data (kmers_5.npy, 50k sample):**

| Model | Spearman r | Pearson r | Top 1 MSE | Top 50 MSE |
|-------|-----------|-----------|-----------|-----------|
| Run 1 (1k) | 0.694 | 0.408 | 0.106 | 0.165 |
| Run 2 (2k) | 0.712 | 0.441 | 0.107 | 0.163 |
| Run 3 (3k) | 0.717 | 0.432 | 0.108 | 0.170 |
| Run 4 (4k) | 0.580 | 0.296 | 0.126 | 0.226 |
| Run 5 (5k) | **0.731** | **0.463** | 0.116 | 0.170 |
| **SFE_SE** | **0.714** | 0.443 | **0.111** | 0.181 |

### Interpretation
- Spearman 0.714 on the common 5k test data — on par with Runs 1-3 despite being trained on 3x fewer samples (4.8M vs 13.4M)
- Only slightly behind Run 5 (0.731), which was trained on the same test distribution with 2.8x more data
- Adding FD + RefSeq diversity improves reconstruction quality (lower MSE) more than retrieval quality (Spearman)
- The model trained on just two environments generalizes well to the full 4-source dataset

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

Tested all 6 models on test data at each threshold (1k, 2k, 3k, 4k, 5k). Each test: 50k samples, 100 queries, 50 neighbors. Spearman correlation (latent distance vs k-mer MSE):

| Model | 1k data | 2k data | 3k data | 4k data | 5k data |
|-------|---------|---------|---------|---------|---------|
| Run 1 (1k) | 0.852 | 0.720 | 0.680 | 0.755 | 0.694 |
| Run 2 (2k) | 0.867 | 0.742 | 0.710 | 0.786 | 0.712 |
| Run 3 (3k) | **0.871** | 0.730 | **0.714** | **0.797** | 0.717 |
| Run 4' (4k) | 0.869 | 0.734 | 0.707 | 0.786 | 0.727 |
| Run 5 (5k) | 0.847 | **0.742** | 0.657 | 0.749 | **0.731** |
| SFE_SE_5 | 0.851 | 0.696 | 0.686 | 0.760 | 0.714 |

### Observations

1. **Shorter test data is easier** — all models score 0.847-0.871 on 1k data vs 0.657-0.731 on 5k data. Shorter sequences have more distinctive k-mer profiles relative to sampling noise.

2. **No model dominates all columns** — column-wise best: 1k→Run 3, 2k→Run 2/5 (tied 0.742), 3k→Run 3, 4k→Run 3, 5k→Run 5. There is no single "best" model across all test conditions.

3. **Run 3 is the best generalist** — highest on 1k (0.871), 3k (0.714), and 4k (0.797), competitive on 2k (0.730) and 5k (0.717). The 3,000 bp threshold provides a good balance of training data quality and quantity.

4. **The diagonal pattern doesn't hold** — models aren't best on their "own" threshold data. Run 3 beats Run 1 on 1k data; Run 3 beats Run 4' on 4k data (0.797 vs 0.786); Run 5 only wins on 5k data.

5. **Run 5 is polarized** — best on 5k (0.731), tied-best on 2k (0.742), but worst on 3k (0.657) and near-worst on 1k (0.847) and 4k (0.749). The strict 5,000 bp threshold loses short-sequence diversity.

6. **4k column has high absolute values** — range 0.749-0.797, second only to 1k. The 4k test set may hit a sweet spot of sequence quality vs diversity.

7. **SFE_SE_5 underperforms on cross-comparison** — despite good per-k-mer reconstruction, it lags the sweep runs on most test sets. Training data diversity (4 sources vs 2) matters more than environment specialization for generalization.

8. **The monotonic trend on 5k data persists** — 0.694 → 0.712 → 0.717 → 0.727 → 0.731, but this is specific to the 5k test condition. On other test sets, the pattern is more complex.

### Sample size stability check (1k data, 100k samples)

Repeated the 1k test with 100,000 samples (vs standard 50,000) to check stability:

| Model | 50k samples | 100k samples | Δ |
|-------|-------------|--------------|-----|
| Run 1 (1k) | 0.852 | 0.858 | +0.006 |
| Run 2 (2k) | 0.867 | 0.874 | +0.007 |
| Run 3 (3k) | 0.871 | 0.873 | +0.002 |
| Run 4' (4k) | 0.869 | 0.874 | +0.005 |
| Run 5 (5k) | 0.847 | 0.852 | +0.005 |
| SFE_SE_5 | 0.851 | 0.848 | -0.003 |

All deltas < 0.01, rankings unchanged. The 50k sample size is sufficient for stable cross-comparison results.

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

Trained Run_SFE_SE_1 through Run_SFE_SE_4 (≥1k through ≥4k, SFE+SE data only) to complement the existing Run_SFE_SE_5. Tested all 6 sweep models on all 5 SFE_SE test datasets. Spearman correlation (50k samples, 100 queries, 50 neighbors):

| Model | SFE_SE_1 | SFE_SE_2 | SFE_SE_3 | SFE_SE_4 | SFE_SE_5 |
|-------|----------|----------|----------|----------|----------|
| Run 1 (1k) | 0.761 | 0.747 | 0.831 | **0.871** | 0.782 |
| Run 2 (2k) | **0.791** | 0.764 | **0.844** | **0.874** | 0.778 |
| Run 3 (3k) | 0.779 | **0.773** | 0.841 | 0.859 | 0.761 |
| Run 4' (4k) | 0.789 | 0.779 | 0.808 | 0.841 | 0.754 |
| Run 5 (5k) | 0.790 | 0.763 | 0.700 | 0.785 | 0.695 |
| SFE_SE_5 | 0.777 | 0.776 | 0.744 | 0.842 | 0.742 |

### Observations

1. **SFE_SE_4 is the easiest test set** — all models score 0.785-0.874, highest column. The 4k aquatic-only data hits a sweet spot.

2. **Run 5 collapses on SFE_SE_3 and SFE_SE_5** — 0.700 and 0.695. Same "polarized" pattern as on mixed data.

3. **Run 2 is best on aquatic data** — wins on SFE_SE_1 (0.791), SFE_SE_3 (0.844), and tied for SFE_SE_4 (0.874). On mixed data, Run 3 was the generalist; on aquatic-only, Run 2 takes that role.

4. **Sweep models beat the aquatic-only model on aquatic data** — Run 1 (0.782), Run 2 (0.778) both outperform SFE_SE_5 (0.742) on its own SFE_SE_5 test data. Training data diversity (4 sources) helps even for aquatic retrieval.

5. **Reversed monotonic trend on SFE_SE_5** — 0.782→0.778→0.761→0.754→0.695. Lower threshold models perform better on the aquatic 5k test data, opposite of the mixed 5k data trend (0.694→0.712→0.717→0.727→0.731).

6. **SFE_SE_1 and SFE_SE_2 are harder than expected** — Spearman 0.747-0.791, lower than the mixed 1k/2k data. With only 2 environments, neighboring sequences are harder to distinguish.

## 2026-02-09: Data shuffling concern and concatenate_matrices fix

### Problem

The VAE uses a contiguous 90/10 train/val split (first 90% train, last 10% val) and assumes pre-shuffled data. The individual source datasets were shuffled internally, but `concatenate_matrices` stacked them in order without shuffling. This means the validation set is dominated by whichever source was concatenated last — creating a systematic distribution shift between train and val sets. This likely explains some of the apparent "overtraining" patterns in the SFE_SE experiments.

### Fix

Added `--shuffle` flag to `concatenate_matrices`. After concatenation, it generates a random permutation and applies it to both the matrix rows and ID lines. Also works with a single input file pair for reshuffling existing datasets.

### Impact

All concatenated datasets used for training should be regenerated with `--shuffle`. Previous train/val loss comparisons may be unreliable due to the distribution shift.
