# Claude Session Notes

## About This File

Torben collaborates with multiple Claude instances:
- **This host (Threadripper)**: Heavy-duty computation - VAE training, large dataset processing, clustering
- **Desktop**: Writing and lighter tasks

These notes are shared across both so each instance can understand context from the other's sessions. Keep notes clear and comprehensive.

## Related Repositories
- **Embedding** (`tnn111/Embedding`, public): This repo — VAE training, inference, k-mer calculation
- **ClusteringPaper** (`tnn111/ClusteringPaper`, private): Nature Methods paper being written by sibling Claude instance on desktop
  - Cloned locally to `/home/torben/ClusteringPaper/`
  - Contains: Introduction.md, Methods_VAE.md, Results_VAE.md, Draft.md, Data.md, References.md, Paper.md
  - Latest commit: `ed11249` — Retarget paper to Nature Methods

## 2026-02-02: Clustering analysis and codebase cleanup

### HDBSCAN Clustering on t-SNE
- Ran HDBSCAN clustering on full 4.8M t-SNE coordinates
- Parameters: `min_cluster_size=1000`, `min_samples=100`
- Discussed `min_samples` parameter: higher = more conservative, denser cores required

### Sample Comparison Methods
Explored three approaches for comparing sample groups (SFE_1_S vs SFE_1_W):
1. **Density difference heatmap**: Normalized 2D histograms, subtract to show enrichment
2. **Cluster composition analysis**: Fraction of each group per HDBSCAN cluster
3. **Overlay plot**: Both groups on same axes (cyan/yellow on black background for green overlap)

### Codebase Rename: VAEMulti → VAE
- Renamed `VAEMulti.py` → `VAE.py`, `VAEMulti.md` → `VAE.md`
- Renamed class `VAEMulti` → `VAE`
- Updated all file paths from `vae_multi_*` → `vae_*`
- Renamed actual model files in `Models/Multi_005_384/`
- Updated references across all documentation and scripts

### Gzip Support for calculate_kmer_frequencies
- Added automatic detection of `.gz` extension
- Uses `gzip.open()` for compressed files, regular `open()` otherwise
- Can mix compressed and uncompressed input files

### Memory Usage Discussion
- `calculate_kmer_frequencies`: Keeps all chunks in memory, concatenates at end
  - Peak: ~2× final array size during concatenation
  - For 4.8M sequences: ~106 GB peak
- Discussed potential optimization: write chunks to disk, use memmap concatenation
- `VAE.py`: Loads all training + validation data into memory (~53 GB for 4.8M sequences)

### NumPy File Handling
- Can read .npy header without loading data: `np.load(file, mmap_mode='r')`
- Discussed potential concatenation utility using memmap to avoid memory issues

### Current Dataset (Data/all_kmers.npy)
- Shape: (4,776,770, 2773)
- Dtype: float32
- Size: 49.35 GB
- All sequences ≥ 5000 bp (min=5000, max=10,208,085)

### VAE Training Parameters (optimal config)
- Latent dim: 384
- β (KL weight): 0.05
- Learning rate: 1e-4
- Batch size: 1024
- Epochs: 500
- Results: Val loss 2899, MSE 1.030, Spearman r=0.95

### In Progress
- Running `calculate_kmer_frequencies` on a larger dataset (at least 3x bigger, 14M+ sequences)
- Planning to train VAE on full larger dataset
- Will compare metrics to 4.8M baseline (val loss 2899, MSE 1.030, 6-mer MSE 1.319)
- Memory estimate: ~160 GB for data, should fit in 512 GB

### Important: Dataset Composition
- **Current 4.8M dataset**: Aquatic/marine environments
- **New larger dataset**: Mostly terrestrial, some aquatic components

This is not just "more data" but **more diverse data**. Terrestrial and aquatic microbial communities have different GC content, codon usage, and k-mer profiles. The latent space may need to accommodate fundamentally different regions of k-mer space.

If metrics degrade, consider:
- Increasing latent dimension (384 → 512 or 768)
- Training separate models for different environments
- The bimodal cosine distance distribution may become multimodal

**Decision**: If the terrestrial data hurts performance, it won't be used for the paper. The paper focuses on aquatic/marine environments, and the current 4.8M model performs well for that purpose. The larger dataset experiment is exploratory.

### Memory Issue Fix (2026-02-02)

Large dataset run failed due to OOM during concatenation. Fixed `calculate_kmer_frequencies`:
- Now writes chunks to temp files instead of keeping in memory
- Uses memmap concatenation at the end
- Memory stays ~1 GB regardless of dataset size
- Temp files cleaned up automatically

### Large Dataset Training Results (2026-02-03)

Training on larger combined dataset (aquatic + terrestrial) completed successfully.

**Final results comparison:**

| Metric | Baseline (4.8M) | Epoch 210 | Final (500) | vs Baseline |
|--------|-----------------|-----------|-------------|-------------|
| Val loss | 2899 | 1953 | **1930** | -33% ✓ |
| MSE | 1.030 | 0.899 | **0.889** | -14% ✓ |
| 6-mer | 1.319 | 1.147 | **1.134** | -14% ✓ |
| 5-mer | 0.203 | 0.191 | **0.187** | -8% ✓ |
| 4-mer | 0.053 | 0.064 | 0.063 | +19% ✗ |
| 3-mer | 0.015 | 0.020 | 0.019 | +27% ✗ |
| 2-mer | 0.008 | 0.009 | **0.008** | same ✓ |
| 1-mer | 0.005 | 0.006 | **0.004** | -20% ✓ |

**Key observations:**
- Shorter k-mers (2-mer, 1-mer) recovered by end of training
- 3-mer and 4-mer still slightly worse than baseline but improved from mid-training
- Overall model significantly better on larger diverse dataset
- Trade-off between k-mer scales is real but partially resolves with more training

**Runtime stats:**
- Time: ~3h 10min (500 epochs)
- Peak memory: 234 GB
- CPU: 187%

### Extended Training with Lower Learning Rate (2026-02-03)

Ran additional 500 epochs with LR 1e-5 (down from 1e-4). Marginal improvement - model essentially converged:

| Metric | After 500 (LR 1e-4) | After 1000 (LR 1e-5) |
|--------|---------------------|----------------------|
| Val loss | 1929.8 | **1927.9** |
| MSE | 0.889 | **0.888** |
| 6-mer | 1.134 | **1.133** |
| Others | unchanged | unchanged |

Model has hit capacity for this architecture. Further gains would require architectural changes.

### Dataset Sizes

- Aquatic dataset: 4,776,770 contigs ≥ 5000 bp
- Terrestrial dataset: 8,006,888 contigs ≥ 5000 bp
- **Combined: ~12.8M sequences**

This is ~1.8x the 7M model parameters. Below traditional 10x guideline but works well due to:
- Strong KL regularization in VAE
- Structured k-mer input (not raw sequences)
- Good convergence observed

### Design Goals

Model designed to be portable:
- **Model size**: ~7M params = ~28 MB (tiny by modern standards)
- **Encoder file**: ~15 MB (`vae_encoder_best.keras`)
- **Inference**: Runs comfortably on laptop CPU, instant on any GPU
- **Training**: Requires powerful machine (data doesn't fit in laptop RAM)

Goal: Train on powerful system, deploy trained encoder anywhere.

### Literature Search (2026-02-03)

Searched for published work on multi-scale k-mer VAEs. **This appears to be relatively uncharted territory.**

**Closest related work:**
- **VAMB** (Nature Biotech 2020) - VAE for metagenomic binning, but uses only 4-mers
- **GenomeFace** (2025) - Uses k=1-10, closest to our approach, but for taxonomic prediction
- **NeurIPS 2024** "Revisiting K-mer Profile for Effective and Scalable Genome Representation Learning" - theoretical analysis, single k values

**Not found in literature:**
- Multi-scale k-mer VAEs with explicit analysis of trade-offs between k-mer sizes
- Studies on how dataset diversity affects reconstruction at different scales
- The specific trade-off phenomenon observed here

**Conclusion**: The trade-off where increased diversity shifts model capacity toward higher-dimensional features (6-mers) at expense of shorter k-mers appears novel and potentially publishable.

### Expanded Dataset Training (2026-02-04)

Added more data to training. Results are dramatically better across all metrics:

| Metric | Previous (12.8M) | New | Improvement |
|--------|------------------|-----|-------------|
| Val loss | 1927.9 | **1546.9** | -20% |
| MSE | 0.888 | **0.614** | -31% |
| 6-mer | 1.133 | **0.793** | -30% |
| 5-mer | 0.187 | **0.094** | -50% |
| 4-mer | 0.063 | **0.025** | -60% |
| 3-mer | 0.019 | **0.005** | -74% |
| 2-mer | 0.008 | **0.002** | -75% |
| 1-mer | 0.004 | **0.001** | -75% |

**Key observations:**
- All metrics improved substantially
- Shorter k-mers (3-mer through 1-mer) improved by ~75% - the previous trade-off resolved
- Model now captures all scales well simultaneously
- The "hit capacity" conclusion from previous training was premature - more diverse data unlocked further improvement

**Runtime stats (from /usr/bin/time):**
- Elapsed: 3:11:35 (500 epochs)
- User time: 20031.06s
- System time: 1065.08s
- CPU: 183%
- Peak memory: 241 GB (240922960 KB maxresident)

**Dataset composition:**
- Aquatic metagenomes: 4,776,770
- Terrestrial metagenomes: 8,006,888
- **NCBI RefSeq representative: 655,640** (new)
- **Total: ~13.4M sequences**

**Critical insight**: Adding only 655K sequences (~5% increase) from RefSeq caused ~30% improvement. Quality and taxonomic diversity matter more than quantity. RefSeq representative genomes provide curated coverage across the tree of life, filling k-mer space gaps that environmental metagenomes miss.

### Extended Training with LR 2e-6 (2026-02-04)

Ran additional 500 epochs with lower learning rate (2e-6). Model continues to improve:

| Metric | After 500 (LR 1e-4) | After 1000 (LR 2e-6) | Change |
|--------|---------------------|----------------------|--------|
| Val loss | 1546.9 | **1517.8** | -1.9% |
| MSE | 0.614 | **0.601** | -2.1% |
| 6-mer | 0.793 | **0.777** | -2.0% |
| 5-mer | 0.094 | **0.089** | -5.3% |
| 4-mer | 0.025 | 0.025 | same |
| 3-mer | 0.005 | **0.004** | -20% |
| 2-mer | 0.002 | 0.002 | same |
| 1-mer | 0.001 | 0.001 | same |

**Runtime stats (from /usr/bin/time):**
- Elapsed: 3:18:48 (500 epochs)
- User time: 20907.88s
- System time: 1199.44s
- CPU: 185%
- Peak memory: 245 GB (256896884 KB maxresident)

Model still improving - not yet converged. Could benefit from more training at low LR.

### New Script: concatenate_matrices (2026-02-04)

Created utility script to concatenate multiple k-mer matrix files and their ID files.

**Usage:**
```bash
./concatenate_matrices \
    -i file1.npy file2.npy file3.npy \
    -id file1.txt file2.txt file3.txt \
    -o combined
```

**Features:**
- Memory-efficient: uses memmap, loads one input file at a time
- Validates row counts in .npy match line counts in corresponding .txt
- Supports gzip-compressed ID files (.txt.gz)
- Output: `<basename>.npy` and `<basename>.txt`

**Implementation:**
- Reads .npy headers without loading data to get shapes and validate
- Creates output memmap with total row count
- Copies matrices one at a time, freeing memory after each
- Peak memory ≈ size of largest input file (not total output)
- Temp chunks now written to local directory (not /tmp) to support parallel runs

### 2026-02-05: Full Codebase Review (Opus 4.6)

Conducted comprehensive review of all project files. Key findings:

**Critical bugs:**
1. `embedding` and `create_and_load_db` use stochastic `z` instead of `z_mean` for inference — adds noise, breaks reproducibility
2. `verify_local_distances.py` has stale column indices (old 7-mer format) — would produce wrong results
3. `convert_txt_to_npy` is outdated (wrong column count, references removed GC column)
4. Root symlink `vae_encoder_best.keras` points to old Dec 3 model, not the Feb 4 best model

**Code quality:**
- `ClipLayer`, `Sampling`, `clr_transform` duplicated across 4 files (divergence already caused bug #2)
- `main.py` is a placeholder leftover from `uv init`
- Models scattered across `Models/`, `Data/`, and project root

**Architecture observations:**
- CLR applied jointly across 6 independently-normalized k-mer groups (theoretically questionable but works empirically)
- No dropout in the architecture
- Training history overwritten on resume (can't plot full multi-run curves)
- Pseudocount 1e-6 creates extreme CLR values for zero-count 6-mers

Full details recorded in VAE.md under "2026-02-05: Full Codebase Review"

**Deferred for future reorganization:**
- #4: Duplicated custom layers across 4 files
- #5: Stale model symlink / models scattered across directories
- #9: Training history not merged across runs

Torben is thinking about how to reorganize the project more systematically. These should be addressed as part of that effort.

**Current training data location:** `../../VAE_Training_Data/all_contigs_l5000.npy`
(The `Data/all_kmers.npy` is the older 4.8M aquatic-only dataset.)

**Verification results (per-group CLR + Jeffreys, epoch ~237, not converged):**
- On training data (50k sample): Spearman r = 0.93, Pearson r = 0.60
- Top 1 MSE = 0.060, random baseline = 0.227
- On old aquatic-only Data/all_kmers.npy: Spearman r = 0.67 (expected — less represented in training)
- Near-convergence runs: Spearman 0.929/0.927 (plateaued), Pearson 0.638/0.620
- Model has converged on this metric; close to previous model's 0.95 with completely different preprocessing

**Planned experiment: minimum contig length sweep**
- Datasets being prepared at 1,000 / 2,000 / 3,000 / 4,000 / 5,000 bp thresholds
- Goal: systematically compare training metrics and local distance quality across cutoffs, choose final threshold
- Now feasible because Jeffreys prior prevents pseudocount-related artifacts for short contigs
- Results intended for inclusion in the Nature Methods paper
- FD (Microflora Danica) paper explicitly filters contigs <3 kbp using SeqKit (v.2.4.0)
  - Consistent with mmlong2 default of 3,000 bp min contig length for binning
  - Also: contigs >250 kbp kept as separate bins, remainder goes to iterative ensemble binning
  - Source: https://github.com/Serka-M/mmlong2
  - Paper: https://www.nature.com/articles/s41564-025-02062-z
  - FD contigs obtained directly from ENA, already filtered at 3 kbp before submission
  - Implication for length sweep: 1k and 2k thresholds only gain shorter contigs from non-FD sources (aquatic, RefSeq)

**Length sweep results (2026-02-08): SUPERSEDED — unshuffled data with biased train/val splits.**
See shuffled retraining results below for current analysis. Key findings that remain valid:
- Jeffreys prior solved the short-contig problem (1,000 bp contigs now viable)
- Reconstruction loss alone is insufficient to evaluate embedding quality
- FD paper confirms 3 kbp minimum is standard (mmlong2 pipeline default)
- ReduceLROnPlateau scheduling can cause reconstruction-representation tradeoff (Run 4 artifact)

**Run_SFE_SE / cross-comparison matrices (2026-02-08/09): SUPERSEDED — unshuffled data.**
- LR schedule: 1st reduction at epoch 248 (between sweep runs and Run 4)
- Key insight: diversity (FD + RefSeq) helps reconstruction more than retrieval quality

**Key insight: reconstruction loss is not enough**
- Run 4 proved that best reconstruction MSE ≠ best embedding quality
- Need independent metrics: Spearman (ranking) and count-vs-distance linearity (geometry)
- The LR scheduling artifact would not have been visible at typical training lengths (100-500 epochs)
- Worth highlighting in the paper: most metagenomic embedding papers only report reconstruction loss
- TODO: re-run count-vs-distance analysis on final model to confirm linear regime persists

**Full cross-comparison matrix (2026-02-08): SUPERSEDED — unshuffled data.** Run_3 was already the best generalist; confirmed with shuffled data (see below).

**SFE_SE cross-comparison (2026-02-09): SUPERSEDED — unshuffled data.**

**Metrics logging bug fix (2026-02-09):**
- "Recon" in logs was computed on 5000-sample val subset, NOT training data
- Created false overtraining signal for SFE_SE_3 (val subset happened to be unrepresentatively easy)
- Fixed: now logs actual Keras training loss as "Train"
- All previous "Recon" values should not be compared to "Val" for overtraining assessment

**Future consideration: float16**
With the Jeffreys prior pseudocounts (smallest is 2.4e-4 for 6-mers), float16 is now feasible — it wasn't with the old 1e-6 pseudocount. Would halve memory for training data (~120 GB → ~60 GB for 13.4M sequences). Revisit during reorganization.

**Data shuffling fix (2026-02-09):**
- Individual source datasets were shuffled, but concatenation stacked them in order
- With contiguous 90/10 split, val set is dominated by the last concatenated source
- Added `--shuffle` flag to `concatenate_matrices` — shuffles both matrix rows and IDs together
- All concatenated training datasets need to be regenerated with `--shuffle`
- Also works with a single file pair for reshuffling existing datasets

### 2026-02-10/11/12: Shuffled retraining sweep — all 5 runs complete

All models retrained on shuffled data (1000 epochs each). Final correlation results (verify_local_distances, 50k samples, own data):

| Run | Spearman | Pearson | Top 1 MSE | Top 50 MSE | Random MSE |
|-----|----------|---------|-----------|-----------|------------|
| Run 1 (1k) | 0.751 | 0.430 | 0.167 | 0.256 | 0.555 |
| Run 2 (2k) | 0.627 | 0.360 | 0.167 | 0.236 | 0.542 |
| **Run 3 (3k)** | **0.721** | **0.388** | **0.119** | **0.194** | 0.511 |
| Run 4 (4k) | 0.697 | 0.412 | 0.122 | 0.194 | 0.494 |
| Run 5 (5k) | 0.511 | 0.242 | 0.139 | 0.240 | 0.468 |

Run 4/5 own-data values updated after kmers_4.npy and kmers_5.npy were replaced.

**Key findings:**
- **Run 3 (3K bp) is the best model** — best own-data Spearman (0.721) and wins cross-threshold comparison on every test condition
- Spearman values are lower than unshuffled runs (e.g., Run 1: 0.852→0.751) because the old unshuffled validation set was biased toward one source
- Run 2 is an unexplained outlier at 0.627; Run 5 dropped to 0.511 after dataset replacement
- Convergence-by-500 pattern confirmed across all runs
- 500 epochs appears sufficient; remaining epochs provide negligible improvement

**Train/val gap — RESOLVED: was a data shuffling artifact, NOT BatchNorm:**

The ~34-48 point train/val gap in Runs 4-5 was caused by **unshuffled data**, not BatchNorm statistics. After the user fixed the shuffling:
- Run 4: gap dropped from 48.1 → ~0.3
- Run 5: gap dropped from 41.8 → ~0.3
- Runs 1-3 never had the gap because their data was already properly shuffled

**ReduceLROnPlateau schedules (from resource.log, NOT vae_training.log):**

Keras writes LR reduction messages to stdout/stderr (captured in `resource.log`). All 5 runs had multiple reductions:

| Run | Start LR | 1st reduction | Floor (1e-6) | Epochs at floor | Reductions |
|-----|----------|---------------|--------------|-----------------|------------|
| Run 1 | 1e-4 | Epoch 21 | Epoch 351 | 649 | 7 |
| Run 2 | 1e-4 | Epoch 21 | Epoch 354 | 646 | 7 |
| Run 3 | 1e-4 | Epoch 22 | Epoch 316 | 684 | 7 |
| Run 4 | 1e-5 | Epoch 416 | Epoch 601 | 399 | 4 |
| Run 5 | 1e-4 | Epoch 248 | Epoch 622 | 378 | 7 |

Runs 1-3 hit floor by epoch 316-354; Runs 4-5 much later (601-622). Run 4 started at lower LR (1e-5), delaying its first reduction. Run 5 had an unusually late first reduction (248 vs 21-22 for Runs 1-3).

**Euclidean vs cosine distance (Run_4 model, updated 4K data):**
- Euclidean Spearman 0.697 vs cosine 0.621 (Δ = -0.076)
- Euclidean wins — VAE's MSE loss creates Euclidean-friendly geometry
- ChromaDB should use `'hnsw:space': 'l2'` instead of `'cosine'`

**Full 5×5 cross-comparison matrix (shuffled data, Run_4/5 retrained after shuffling fix):**

| Model \ Test | 1K | 2K | 3K | 4K | 5K | Mean |
|---|---|---|---|---|---|---|
| Run 1 (1k) | **0.751** | 0.616 | 0.723 | 0.703 | 0.635 | 0.686 |
| Run 2 (2k) | 0.764 | **0.627** | 0.729 | 0.711 | 0.643 | 0.695 |
| **Run 3 (3k)** | **0.769** | **0.639** | **0.721** | **0.722** | **0.660** | **0.702** |
| Run 4 (4k) | 0.738 | 0.598 | 0.692 | 0.674 | 0.625 | 0.665 |
| Run 5 (5k) | 0.726 | 0.584 | 0.655 | 0.640 | 0.616 | 0.644 |

Run_3 wins every column. Mean: R3 (0.702) > R2 (0.695) > R1 (0.686) > R4 (0.665) > R5 (0.644). Clear tier structure: Tier 1 (R3), Tier 2 (R1-R2), Tier 3 (R4-R5).

**Top 1 MSE matrix (lower = better):**

| Model \ Test | 1K | 2K | 3K | 4K | 5K | Mean |
|---|---|---|---|---|---|---|
| Run 1 (1k) | 0.167 | 0.172 | 0.108 | 0.117 | 0.105 | 0.134 |
| Run 2 (2k) | 0.175 | 0.167 | 0.113 | 0.126 | 0.103 | 0.137 |
| **Run 3 (3k)** | **0.160** | **0.156** | **0.119** | **0.121** | 0.103 | **0.132** |
| Run 4 (4k) | 0.178 | 0.176 | 0.127 | 0.138 | 0.101 | 0.144 |
| Run 5 (5k) | 0.180 | 0.176 | 0.125 | 0.140 | 0.100 | 0.144 |

MSE confirms Spearman ranking. 1K/2K test data has higher MSE (~0.16-0.18) than 3K-5K (~0.10-0.14) — shorter sequences have noisier k-mer profiles.

**Remaining:** SFE_SE runs still pending.

**ClusteringPaper repo updated** to commit 97a70ac (pulled 2026-02-12).
