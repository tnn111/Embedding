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
- Epochs: 1000 (converges by ~500)
- Per-group CLR with Jeffreys prior pseudocount
- Best model: Run_3 (3K bp threshold), mean Spearman 0.702 on augmented data

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
| Run 4 | 1e-4 | Epoch 22 | Epoch 468 | 532 | 7 |
| Run 5 | 1e-4 | Epoch 21 | Epoch 346 | 654 | 7 |

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

**Run_5 2-mer/1-mer anomaly**: Run_5 training log shows higher 2-mer (0.000450) and 1-mer (0.000135) MSE than Runs 2-4, despite lower total MSE. Root cause: extreme high-GC sequences (>75% GC) are underrepresented in ≥5K bp training data. Only 46 val samples (0.46%) but their 1-mer MSE is 26x higher than Run_3's (0.0099 vs 0.0004), contributing ~50% of total 1-mer MSE. Not a noise or capacity issue — purely training data coverage at GC distribution tails. 3K data retains more extreme-GC organisms (4 GC peaks vs 3 for 5K), further supporting Run_3 as best general-purpose encoder.

**SFE_SE cross-comparison (2026-02-13):**

SFE_SE models on **augmented test data** (kmers_1-5):

| Model \ Test | 1K | 2K | 3K | 4K | 5K | Mean |
|---|---|---|---|---|---|---|
| SFE_SE_1 (1K) | 0.907 | 0.820 | 0.881 | 0.843 | 0.829 | **0.856** |
| SFE_SE_2 (2K) | 0.889 | 0.808 | 0.873 | 0.827 | 0.807 | 0.841 |
| SFE_SE_3 (3K) | 0.886 | 0.802 | 0.868 | 0.813 | 0.811 | 0.836 |
| SFE_SE_4 (4K) | 0.878 | 0.804 | 0.850 | 0.807 | 0.805 | 0.829 |

SFE_SE models on **SFE_SE test data** (kmers_SFE_SE_1-5):

| Model \ Test | SE_1 | SE_2 | SE_3 | SE_4 | SE_5 | Mean |
|---|---|---|---|---|---|---|
| SFE_SE_1 (1K) | 0.773 | 0.739 | 0.723 | 0.816 | 0.779 | 0.766 |
| SFE_SE_2 (2K) | 0.743 | 0.714 | 0.682 | 0.784 | 0.758 | 0.736 |
| SFE_SE_3 (3K) | 0.749 | 0.704 | 0.676 | 0.776 | 0.736 | 0.728 |
| SFE_SE_4 (4K) | 0.751 | 0.684 | 0.687 | 0.781 | 0.726 | 0.726 |
| SFE_SE_5 (5K) | 0.862 | 0.862 | 0.823 | 0.868 | 0.819 | **0.847** |

Key findings:
- All SFE_SE models dramatically outperform augmented runs (worst SFE_SE 0.829 > best augmented 0.702 on augmented data)
- On augmented data: SFE_SE_1 wins (0.856). On SFE_SE data: SFE_SE_5 wins (0.847) — opposite ranking
- SFE_SE data is harder: models 1-4 score 0.73-0.77 vs 0.83-0.86 on augmented data
- SFE_SE_5 dominates every column on SFE_SE data (+0.081 over second place)

**ClusteringPaper repo updated** to commit 97a70ac (pulled 2026-02-12).

### 2026-02-14: Notebook exploration of SFE_SE embeddings

Working through `clustering.ipynb` with new SFE_SE_1 embeddings (6,693,829 sequences, SFE_SE_5 model).

**Pairwise distance distributions (10K sample):**
- Euclidean: range 2.67–69.98, mean 27.33, std 7.51 — unimodal, longer right tail
- Cosine: range 0.025–1.334, mean 1.000, std 0.061 — unimodal, symmetric (was bimodal with old 4.8M data)

The cosine distribution collapsing to unimodal around 1.0 confirms Euclidean is the right metric — cosine has almost no discriminative range (std 0.061) in 384-dim space. Random high-dimensional vectors tend toward orthogonality, so cosine can't distinguish structure that Euclidean preserves.

One-vs-all cosine from random sequence (SFE_2_S_c_194575 vs 6.7M others): mean 1.001, std 0.060 — same shape, no bimodal structure. Old bimodality was likely from mixed marine/terrestrial data.

**Intrinsic dimensionality of latent space — TWO-NN estimate: d̂ ≈ 9:**

Applied the TWO-NN estimator (Facco et al. 2017) to 100K sampled embeddings using Euclidean distance. Result: d̂ = 9.12. The 384-dimensional latent space encodes data on a ~9-dimensional manifold. This is plausible — multiple independent axes of variation (GC content, genome size, taxonomic lineage, codon usage, etc.) each contribute a dimension.

The count-vs-distance plot (cell 9) appeared to show a linear regime at small distances, which naively suggests d ≈ 2 (since the NN distance PDF goes as r^(d-1) at small r). But this was a visual artifact: the early part of an r^8 curve looks approximately linear over a narrow range. The TWO-NN estimator is more reliable because it uses the ratio μ = r₂/r₁ which is density-independent.

Results so far (100K sample, Euclidean distance):
- Manual TWO-NN (float32): d̂ = 9.12 — WRONG, float32 corrupts μ ratios
- Manual TWO-NN (float64): d̂ = 7.61 — pulled down by extreme outliers (μ max = 36M, mean = 725)
- scikit-dimension TWO-NN (10K, float64 internally): d̂ = 66.02
- scikit-dimension MLE (10K): d̂ = 58.94
- scikit-dimension lPCA (10K): d̂ = 284 — failure (too few points for 384-d)
- scikit-dimension DANCo (10K): d̂ = 384 — failure (returned ambient dimension)

The huge discrepancy between our manual TWO-NN (~8) and scikit-dimension's (~66) is due to outlier handling. Our fit-through-origin is dominated by extreme μ values from singletons (isolated sequences with no close relatives). scikit-dimension likely trims or handles these differently.

**TODO: Re-examine intrinsic dimensionality after building ChromaDB database.** Singletons (rare organisms with no family in the dataset) violate the TWO-NN local uniformity assumption. Marine metagenomes (SFE/SE) are expected to produce significant numbers of singletons due to the "rare biosphere" — low-abundance uncultured lineages that appear only once. This could be a substantial fraction of the 6.7M sequences. Easier to filter these once we can query neighbors via ChromaDB. Could filter by nn1 distance percentile before fitting, or use GRIDE (Denti et al. 2022) which is more robust to noise by using higher-order neighbor ratios.

References:
- Facco, E., d'Errico, M., Rodriguez, A. & Laio, A. "Estimating the intrinsic dimension of datasets by a minimal neighborhood information." Scientific Reports 7, 12140 (2017). https://doi.org/10.1038/s41598-017-11873-y
- Denti, F. et al. "The generalized ratios intrinsic dimension estimator." Scientific Reports 12, 20005 (2022). https://doi.org/10.1038/s41598-022-20991-1
- scikit-dimension package: https://github.com/scikit-learn-contrib/scikit-dimension

**ChromaDB updated (2026-02-14):**
- `create_and_load_db` now supports pre-computed embeddings via `-emb` flag (skips encoder/CLR)
- Distance metric changed from cosine to L2 (Euclidean)
- Usage: `./create_and_load_db -id Runs/ids_SFE_SE_1.txt -emb Runs/embeddings_SFE_SE_1.npy`

**Notebook memory optimization (2026-02-14):**

The notebook was crashing the kernel due to excessive memory usage. Key fixes:
- Replaced all scipy `pdist`/`cdist` calls with float32 numpy matrix operations (dot products for cosine, squared-distance decomposition for Euclidean) — scipy upcasts to float64
- Replaced 56 GB `six_mers.flatten()` with chunked histogram computation
- Replaced 80 GB `squareform` (100K×100K pairwise matrix) with chunked nearest-neighbor search
- Pre-computed `embeddings_normed` once in cell 0 for reuse across cosine distance cells
- Switched 100K nearest-neighbor analysis from cosine to Euclidean distance (consistent with finding that Euclidean > cosine for this VAE)
- Switched t-SNE from cosine to Euclidean metric; ran on full 6.7M embeddings (~36 min)

**t-SNE structure (Euclidean, 6.7M points):** Shows distinct "plaques" surrounded by empty space ("moats"). Torben suspects this is where communities will be found — each plaque is a group of organisms with similar k-mer signatures. Leiden community detection should map onto these plaques. This plaque/moat structure also complicates global intrinsic dimensionality estimation (within-plaque dimension is likely low, but the discrete structure between plaques confuses global estimators).

**HDBSCAN on t-SNE (Euclidean, 6.7M points):** 7,391 clusters, 42.2% noise. High noise fraction expected — marine metagenomes contain many "genomic corpses" (degraded DNA from dead cells, viral fragments, free environmental DNA) that produce contigs with incoherent k-mer profiles. These scatter as singletons in the moats. Leiden clustering with a kNN distance threshold should naturally exclude these by requiring minimum connectivity.

**GC-discordant spots in t-SNE:** When colored by GC content, the high-GC regions contain small spots of lower GC, and vice versa. This shows the VAE embeds by higher-order k-mer patterns (6-mer, 5-mer) reflecting phylogeny/codon usage, not just base composition. Species with atypical GC for their lineage end up near their taxonomic neighbors rather than with GC-similar but unrelated organisms. Evidence that multi-scale k-mer input captures signal beyond simple GC content. These GC-discordant spots may split into sub-communities under Leiden clustering.

### kNN graph construction (2026-02-14)

Created `query_neighbors` script — PEP 723 standalone tool that queries ChromaDB for k nearest neighbors and outputs a flat TSV file.

**Output format:** `query_id<TAB>neighbor1(dist1)<TAB>neighbor2(dist2)<TAB>...`
- Distances are Euclidean (sqrt of ChromaDB's squared L2)
- Neighbors ordered nearest-first
- Default k=15 (configurable via `-k`)

**Usage:** `./query_neighbors -id ids.txt -emb embeddings.npy -k 15 > neighbors.tsv`

This intermediate file enables:
- Distance distribution analysis (singleton identification by nn1 distance)
- kNN graph construction for Leiden community detection
- Intrinsic dimensionality re-estimation after singleton filtering
- General exploration of neighborhood structure

**Ran with k=50:** `./query_neighbors -id Runs/ids_SFE_SE_1.txt -emb Runs/embeddings_SFE_SE_1.npy > Runs/neighbors_SFE_SE_1.tsv`
- 6,693,829 rows, 7.5 GB, completed overnight
- Default updated to k=50 (query once, cut to reduce)

### Leiden Community Detection Plan (2026-02-15)

**Phase 1: Distance landscape** (quick, informative, guides all downstream decisions)

Extract the nn1 distance for every sequence and plot the distribution. This tells us:
- Where the natural "singleton" threshold is (if bimodal, the valley is the cutoff)
- The overall distance scale and density variation
- What fraction of the 6.7M sequences are genomic corpses vs connected

Also look at the nn1/nn2 ratio distribution — sequences with nn1 ≈ nn2 are embedded in a dense neighborhood; sequences with nn1 ≪ nn2 are tight pairs or small groups with a moat around them.

**Phase 2: Build a weighted graph for Leiden**

From the k=50 lists, construct a **mutual kNN graph with SNN weights**:
- **Mutual kNN (k=15)**: edge between i and j only if j is in i's 15-NN *and* i is in j's 15-NN. This removes asymmetric connections — if A thinks B is close but B has much closer neighbors, that's not a real community link.
- **SNN weight**: for each mutual edge, weight = number of shared neighbors in their k=15 lists. This rewards pairs embedded in the same dense neighborhood and naturally down-weights connections to singletons.

Singletons will have zero or very few mutual connections and fall out naturally — no need for a hard threshold.

**Phase 3: Leiden community detection**

Run Leiden on the SNN-weighted mutual kNN graph. Key decisions:
- Resolution parameter controls granularity. Start with a moderate value, then sweep.
- The quality function (modularity vs. CPM) affects how resolution maps to cluster sizes.

**Phase 4: Evaluate and characterize**

- Number of communities, size distribution, noise fraction
- Overlay community labels on the t-SNE — do they correspond to the plaques?
- Characterize communities by GC content, source (SFE vs SE), sequence length
- Re-estimate intrinsic dimensionality within the largest communities (where the local uniformity assumption holds)

### Phase 1 Results: Distance Landscape (2026-02-15)

**Data:** 6,693,829 sequences, k=50 neighbors each from ChromaDB (Euclidean/L2).

**nn1 distances:** Range 0.0000–52.62, mean 16.55, median 15.87. Smooth unimodal distribution — no bimodal gap separating connected from isolated sequences. Percentiles: P10=6.55, P50=15.87, P90=27.25, P95=30.48, P99=37.20. About 10% of sequences have nn1 > 27 (most isolated, likely genomic corpses in moats).

**nn1–nn50 spread:** Remarkably tight. Mean nn1=16.55, mean nn50=18.36 (difference ~1.8). Neighborhoods are compact: if nn1≈10, nn50≈12. Points cluster tightly along a line slightly above the nn50=nn1 diagonal.

**μ = nn2/nn1 ratio:** Median 1.006 — for 98%+ of sequences, nn2 ≈ nn1 (dense, uniform neighborhoods). Only 1.84% have μ > 2 (pairs/small isolated groups), 0.45% have μ > 5 (strongly isolated). 974 sequences (0.015%) have nn1=0 (identical embeddings — biologically legitimate, e.g. same-species contigs from different samples).

**Interpretation:** The latent space is well-structured for community detection. Dense plaques with tight neighborhoods, connected by moats of more isolated sequences. No hard singleton threshold needed.

### Phase 2: Graph Construction (2026-02-15)

**Mutual kNN failed — too restrictive for this data:**

| k | Mutual edges | Connected nodes | Isolated |
|---|---|---|---|
| 15 | 1,816,191 | 975,459 (14.6%) | 85.4% |
| 30 | 3,144,013 | 1,055,690 (15.8%) | 84.2% |

Increasing k from 15 to 30 barely changed the isolation rate (85.4% → 84.2%). We gained more edges between already-connected nodes but almost no new nodes. The edges that do exist are very strong (k=30: mean SNN weight 20.4/30, median 21).

**Root cause:** Most sequences have asymmetric nearest-neighbor relationships. Sequences in sparse regions point toward dense clusters, but cluster members don't point back because they have closer neighbors within the cluster. In a latent space with intrinsic dimensionality ~9-60 and wide density variation (nn1 range 0-53), the mutual requirement eliminates the vast majority of edges.

**For comparison:** HDBSCAN on t-SNE found 42% noise. Mutual kNN found 84-85% isolated — almost double. The extra ~42% are sequences that belong to clusters but sit at cluster boundaries or in density gradients where mutual connections don't form.

**Also built symmetric kNN graph for comparison:**
- Edge (i,j) exists if j ∈ kNN(i) OR i ∈ kNN(j) (union, not intersection)
- Weight = |kNN(i) ∩ kNN(j)| (shared nearest neighbors)
- Every node is guaranteed at least k edges (to its k-NN)
- Truly isolated sequences will have low SNN weights; Leiden handles this via resolution parameter
- Closer to what Scanpy does (UMAP-style fuzzy simplicial set also guarantees minimum connectivity)

**Important caveat: mutual kNN may not have "failed."** The 85% isolation rate could be biologically correct:
- Marine metagenomes are expected to have massive numbers of singletons (genomic corpses, rare biosphere)
- HDBSCAN found 42% noise on 2D t-SNE — in the full 384-dim space, even more sequences may be truly isolated
- The mutual requirement is biologically meaningful: if A points to B but B has much closer relatives, A isn't truly part of B's community
- The 975K connected nodes with 1.8M strong edges (mean SNN 9.55/15) may be exactly the real communities — high-confidence connections only

**Plan: Run Leiden on BOTH graphs and compare.** Mutual graph for clean, high-confidence communities; symmetric graph for broader coverage. The mutual graph could give more biologically meaningful results if most "isolated" sequences really are isolated.

**Symmetric graph results (k=15):**

| | Mutual (k=15) | Symmetric (k=15) |
|---|---|---|
| Edges | 1,816,191 | 98,591,244 |
| Connected nodes | 975,459 (14.6%) | 6,693,829 (100%) |
| Mean SNN weight | 9.55 | 2.90 |
| Median weight | 10 | 2 |
| Weight = 0 | 11,100 (0.6%) | 26,811,404 (27.2%) |
| Weight >= 5 | — | 24,264,805 (24.6%) |
| Weight >= 10 | — | 5,537,050 (5.6%) |

The ~5.5M edges with weight >= 10 in the symmetric graph correspond roughly to the mutual kNN core. The other ~93M edges are weaker asymmetric connections from sparse regions pointing toward dense clusters. 27% of edges have weight 0 (no shared neighbors at all). Good structure for Leiden — strong edges form tight communities, resolution parameter controls whether weakly-connected sequences get pulled in.

### Phase 3: Leiden Community Detection (2026-02-15)

**Scripts created:**
- `build_snn_graph` — Builds symmetric kNN graph with SNN weights from neighbors TSV (PEP 723 standalone)
- `leiden_cluster` — Runs Leiden community detection on SNN-weighted graph (PEP 723 standalone)
- `verify_knn_quality` — Evaluates embedding quality using true kNN from ChromaDB (PEP 723 standalone)

**verify_knn_quality results (SFE_SE_5 model, 100K queries, 50 neighbors each):**
- Per-query Spearman (latent rank vs CLR k-mer rank): mean 0.2558, median 0.2112, std 0.1936
- Top-1 MSE: 0.2258, Top-50 MSE: 0.2645, Random baseline MSE: 0.5905
- Not comparable to verify_local_distances Spearman (~0.77-0.85): that measures global distance correlation in a random subsample; this measures fine-grained local ranking fidelity among true nearest neighbors. Lower values expected because local ranking is a much harder task.

**Leiden results (symmetric kNN, k=15, resolution=1.0, 10 iterations):**

| Metric | 2 iterations | 10 iterations |
|---|---|---|
| Communities | 11,137 | 11,180 |
| Singletons | 9,444 (0.1%) | 9,444 (0.1%) |
| Non-singleton communities | 1,693 | 1,736 |
| Median community size | 43 | 45 |
| Mean community size | 3,879 | 3,851 |
| Largest community | 737,449 | 737,449 |

The 2-iteration and 10-iteration results are nearly identical — the algorithm had already largely converged after 2 iterations. 43 more communities appeared (likely from splitting a few marginal cases), but the overall structure is unchanged.

**Top 20 communities (10-iteration):**

| Rank | Community | Size |
|---|---|---|
| 1 | C0 | 737,449 |
| 2 | C1 | 698,921 |
| 3 | C2 | 336,008 |
| 4 | C3 | 330,097 |
| 5 | C4 | 282,403 |
| 6 | C5 | 223,276 |
| 7 | C6 | 199,614 |
| 8 | C7 | 129,185 |
| 9 | C8 | 128,271 |
| 10 | C9 | 104,892 |
| 11-20 | C10-C19 | 62,776–95,132 |

Top 2 communities alone contain 21.4% of all sequences. Top 20 contain ~55%. Community size distribution is heavy-tailed — a few massive communities plus a long tail of thousands of small ones.

**Observations:**
- 99.9% of sequences assigned to non-singleton communities (only 9,444 singletons out of 6.7M)
- The low singleton rate contrasts sharply with HDBSCAN's 42% noise — Leiden with symmetric kNN guarantees connectivity, so even isolated sequences get assigned to some community
- Whether this is appropriate depends on the use case: for taxonomy, assigning genomic corpses to communities may add noise; for survey/coverage, it ensures nothing is lost
- Higher resolution would split the massive top communities into more biologically meaningful groups
- Weight-0 edges (27.2% of graph) were dropped via `--min-weight 1`; only SNN-weighted edges used

**Cell 20 (t-SNE overlay):** Top 20 communities colored distinctly on t-SNE, rest in grey. Communities map well onto the plaque structure visible in the t-SNE.

### Symmetric Leiden results biologically meaningless (2026-02-15)

The symmetric kNN Leiden results produced mega-communities (737K sequences in C0) spanning both Baltic Sea and San Francisco Estuary — not biologically meaningful. Even sampling 5 IDs from C0 showed a mix of SE and SFE sources.

### Mutual kNN Leiden (2026-02-15)

Switched to mutual kNN graph and ran Leiden with multiple configurations:

| Config | Singletons | Communities | Largest | Median size |
|---|---|---|---|---|
| Mutual k=15, SNN count | 85.6% | 191,113 | 3,092 | 2 |
| Mutual k=30, SNN count | 84.4% | 170,374 | 4,503 | 2 |
| Mutual k=30, SNN×exp(-d/15.87) | 84.4% | 170,400 | 4,516 | 2 |
| Mutual k=30, SNN/(d+0.1) | 84.4% | 170,436 | 4,516 | 2 |

**Key finding: edge weights don't matter.** All three weighting schemes for k=30 produced nearly identical results. The community structure is determined by graph topology (which edges exist), not edge weights. With mutual kNN, the graph is already sparse enough that Leiden has little freedom in partitioning.

**Distance-weighted SNN:** Added `--distance-weighted` and `--delta` flags to `build_snn_graph`. Two schemes implemented:
1. `SNN × exp(-dist/scale)` — exponential decay, scale auto-computed as median nn1
2. `SNN / (dist + δ)` — hyperbolic decay, δ=0.1 default

The hyperbolic scheme is theoretically better: identical k-mers get weight SNN/δ (very high), and in a beta-VAE only local distances are meaningful, so 1/dist correctly amplifies close pairs. But in practice neither changed Leiden results.

**`leiden_cluster` updated:** now reads float weights (was int). `--min-weight` default changed to 0.0.

### Neighborhood Growth Analysis (2026-02-15)

**Critical finding: the latent space has two distinct populations.**

Analyzed how many of each sequence's 50 nearest neighbors fall within distance d. The neighborhood growth function is nearly a step function — for any given sequence, either ALL 50 neighbors are close or NONE are.

At d=10 (100K sample): 77% have 0 neighbors, spike at 50 (all neighbors close). Very little in between. Strongly bimodal.

| d | P10 | P25 | P50 | P75 | P90 | Mean |
|---|---|---|---|---|---|---|
| 5 | 0 | 0 | 0 | 0 | 0 | 0.6 |
| 10 | 0 | 0 | 0 | 0 | 50 | 6.9 |
| 15 | 0 | 0 | 0 | 50 | 50 | 19.3 |
| 20 | 0 | 0 | 50 | 50 | 50 | 31.3 |

**Implication:** A simple distance threshold (around 8-12) on nn1 naturally separates sequences with dense local neighborhoods from isolated sequences. Connected components at that threshold ARE the clusters — no need for SNN weights, Leiden, or resolution tuning. The data has a natural gap.

This matches Torben's earlier manual exploration of the data.

**Full dataset results (6,693,829 sequences):** Confirms and sharpens the 100K sample. The step-function behavior is even cleaner:

| d | P10 | P25 | P50 | P75 | P90 | Mean |
|---|---|---|---|---|---|---|
| 5 | 0 | 0 | 0 | 0 | 0 | 0.6 |
| 8 | 0 | 0 | 0 | 0 | 2 | 3.4 |
| 10 | 0 | 0 | 0 | 0 | 50 | 6.9 |
| 12 | 0 | 0 | 0 | 5 | 50 | 11.5 |
| 15 | 0 | 0 | 0 | 50 | 50 | 19.3 |
| 20 | 0 | 0 | 50 | 50 | 50 | 31.4 |
| 25 | 0 | 50 | 50 | 50 | 50 | 40.9 |
| 30 | 50 | 50 | 50 | 50 | 50 | 46.6 |

At d=10, the jump from P75=0 to P90=50 is absolute — sequences either have zero or all 50 of their nearest neighbors within that radius. The mean (6.9) is an average of zeros and fifties, not a typical value.

**Key paper finding:** Only a small fraction of metagenomic sequences are clusterable based on sequence characteristics. The majority (~75-80%) sit in sparse, isolated regions of the latent space with no close neighbors. This isn't a limitation of the embedding — it reflects genuine biological diversity in marine metagenomes. The "rare biosphere" (low-abundance, uncultured lineages) and genomic corpses (degraded environmental DNA) produce sequences with unique k-mer signatures that don't group with anything else. Only the minority with dense neighborhoods represent well-sampled, coherent taxonomic groups amenable to sequence-based clustering.

### Distance-threshold clustering at d=10 (2026-02-15)

**Connected components (d < 10):**
- 42,051,700 edges
- 74.8% singletons (5,007,254 sequences)
- 25.2% clustered (1,686,575 sequences)
- 121,492 non-singleton components
- **Giant component problem:** Largest component has 1,297,780 sequences (77% of all clustered). Second largest is only 882. Transitivity chains merge distant sequences through intermediaries.

Running Leiden on the d<10 graph to split the giant component.

### Key insight: the latent space is an archipelago, not a continuum (2026-02-15)

The distance-threshold analysis reveals why adding more diverse training data (terrestrial metagenomes, NCBI RefSeq) improved reconstruction loss but did not proportionally improve embedding quality (Spearman correlation) for marine sequences.

The latent space is not a continuum — it is an archipelago. Each species or lineage occupies its own tiny, dense "shell" surrounded by empty space. 74.8% of sequences have no neighbor within Euclidean distance 10. The remaining 25.2% that are clusterable represent well-sampled lineages where multiple sequences from the same or closely related organisms fall within each other's neighborhoods.

Adding more species from different environments (terrestrial, RefSeq) does not make existing clusters denser or tighter. It creates more isolated shells in the vast 384-dimensional space. The VAE faithfully learns to reconstruct each new species in its own region (lower MSE), but none of those new shells are close enough to existing marine sequences to improve their neighborhood structure. More training data = more islands, not denser islands.

This explains several earlier findings:
1. **SFE_SE models beat augmented models** even on augmented test data (SFE_SE_1 Spearman 0.856 vs Run_3's 0.702). Focused marine training creates denser shells for marine lineages.
2. **SFE_SE_5 dominates on SFE_SE data** (0.847 vs 0.766 for SFE_SE_1). Deeper sampling of marine lineages at ≥5 kbp means more sequences per shell, making the clusterable fraction more meaningful.
3. **Loss improved but Spearman didn't track**: Adding 655K RefSeq sequences caused ~30% MSE improvement but the local neighborhood structure for marine sequences was unchanged — those new species just scattered as singletons.
4. **The bimodal neighborhood structure**: Sequences either have ALL 50 neighbors close (dense shell) or NONE close (isolated). There is no gradual transition. This is a property of biological sequence diversity, not the embedding method.

The practical implication for metagenomic analysis: sequence-based clustering only works for the well-sampled fraction of a community. The majority of metagenomic sequences represent the "rare biosphere" — low-abundance, uncultured lineages that appear as singletons regardless of embedding quality. The only way to improve clustering coverage for a given environment is deeper sampling of that environment (more sequences from the same species), not broader taxonomic diversity in training.

This is a key finding for the paper. It reframes the VAE not as a tool that clusters everything, but as a tool that reveals which fraction of a metagenome is clusterable — and demonstrates that this fraction is determined by sampling depth, not by the embedding method.

### Leiden on d<10 graph results (2026-02-15)

**Connected components had a giant component problem:** 1,297,780 sequences (77% of clustered) merged into one component via transitivity chains. Ran Leiden (resolution=1.0, unweighted) to split it.

| | Connected Components | Leiden |
|---|---|---|
| Largest | 1,297,780 | 113,878 |
| 2nd largest | 882 | 74,301 |
| Top 20 range | 264–1,297,780 | 16,015–113,878 |
| Singletons | 74.8% | 74.8% (unchanged) |
| Non-singleton communities | 121,492 | 122,175 |

Top 20 communities: 113,878 / 74,301 / 37,159 / 33,243 / 33,094 / 31,700 / 28,152 / 27,421 / 26,711 / 26,666 / 24,398 / 22,917 / 22,710 / 21,702 / 19,519 / 17,954 / 17,900 / 16,478 / 16,058 / 16,015

**t-SNE overlay observations:**
- Each colored top-20 community sits in a distinct, compact region — no splattering across the map
- Light blue (non-top-20 communities) fills the smaller plaques throughout the space
- Grey singletons form the "moat" between plaques — the diffuse background
- Communities don't overlap each other on t-SNE, confirming they correspond to genuine local structure in the high-dimensional space
- The largest community (C0, 113K) occupies one of the major central plaques
- This is the archipelago structure made visible: dense islands of related sequences separated by empty space, with the vast singleton majority in the gaps

**Community coverage table (top N communities by size):**

| Top N | Sequences | % Total | % Clustered | Smallest |
|---|---|---|---|---|
| 10 | 432,325 | 6.5% | 25.6% | 26,666 |
| 20 | 627,976 | 9.4% | 37.2% | 16,015 |
| 30 | 759,835 | 11.4% | 45.1% | 10,252 |
| 40 | 846,954 | 12.7% | 50.2% | 7,574 |
| 50 | 911,887 | 13.6% | 54.1% | 5,749 |
| 100 | 1,104,897 | 16.5% | 65.5% | 2,319 |
| 200 | 1,231,518 | 18.4% | 73.0% | 734 |
| 500 | 1,312,725 | 19.6% | 77.8% | 93 |
| 1000 | 1,336,803 | 20.0% | 79.3% | 29 |

Coverage flattens hard after ~200 communities. Going from 200 to 1000 only gains 6 percentage points. The remaining ~21% of clustered sequences are spread across 121K tiny communities (pairs, triplets). Top 200 is the natural sweet spot.

**Assessment:** The distance threshold at d=10 does the heavy lifting (deciding what's clusterable), and Leiden handles the internal community structure. Much more convincing than pure kNN Leiden. On the right track but not final — may need resolution tuning, community characterization, or threshold adjustment.

**Next steps:**
- Characterize communities (GC, source, length, taxonomy)
- Try different resolution parameters to control granularity
- Consider whether d=10 is optimal or if d=8 or d=12 would be better

## 2026-02-15: leiden_sweep script

Created `leiden_sweep` — PEP 723 standalone script for sweeping distance thresholds on kNN graphs with Leiden clustering.

### Design
- **v1 (abandoned):** Parse all 335M edges into sorted numpy arrays, then searchsorted per threshold. Failed — building 335M Python list elements took 80+ min due to pure-Python overhead.
- **v2 (current):** Re-parse TSV per threshold, breaking early since neighbors are distance-sorted. d=4 parses in 14s (1.5M edges), d=12 will be slower but each threshold is independent.
- Summary stats to stdout as TSV; optional per-threshold community files via `--save-dir`

### CLI
```
./leiden_sweep -n neighbors.tsv -id ids.txt --start 4 --stop 12 --step 0.5 --save-dir sweep/
```

### Output columns
threshold, n_edges, n_singletons, pct_singletons, n_communities, n_nonsingleton, largest, top10_sizes, seqs_in_top200, pct_clustered_in_top200, modularity, median_nonsingleton_size, elapsed_seconds

### Full sweep results (SFE_SE_1, resolution=1.0, seed=42)

Sweep completed in ~14 hours. Community files saved in `Runs/sweep/`.

| d | edges | sing% | communities | largest | top 200 coverage |
|-----|-------|-------|-------------|---------|-----------------|
| 4.0 | 946K | 96.0% | 66,742 | 1,495 | 19.9% |
| 4.5 | 1.6M | 94.9% | 78,746 | 3,518 | 23.4% |
| 5.0 | 2.6M | 93.6% | 89,579 | 7,228 | 27.9% |
| 5.5 | 4.1M | 92.2% | 98,837 | 11,154 | 32.9% |
| 6.0 | 5.9M | 90.6% | 106,465 | 16,403 | 38.1% |
| 6.5 | 8.4M | 88.9% | 112,547 | 21,717 | 43.7% |
| 7.0 | 11.4M | 87.1% | 117,210 | 31,024 | 48.6% |
| 7.5 | 15.1M | 85.2% | 120,690 | 38,776 | 53.7% |
| 8.0 | 19.3M | 83.2% | 122,806 | 48,902 | 58.4% |
| 8.5 | 24.2M | 81.2% | **123,853** | 60,352 | 62.2% |
| 9.0 | 29.6M | 79.1% | **123,877** | 73,375 | 66.5% |
| 9.5 | 35.6M | 77.0% | 123,223 | 89,956 | 70.2% |
| 10.0 | 42.1M | 74.8% | 122,177 | 109,151 | 72.9% |
| 10.5 | 49.0M | 72.6% | 120,423 | 132,590 | 76.2% |
| 11.0 | 56.3M | 70.3% | 118,188 | 157,166 | 78.9% |
| 11.5 | 64.1M | 68.1% | 115,616 | 181,023 | 81.0% |
| 12.0 | 72.2M | 65.8% | 112,969 | 215,677 | 83.1% |

Key observations:
- **Community count peaks at d=8.5–9.0** (~123.9K non-singleton communities), then declines as communities merge at higher thresholds
- **Largest community grows steadily** from 1.5K (d=4) to 216K (d=12)
- **Singleton rate** drops roughly linearly from 96% to 66%
- **Top 200 coverage** (fraction of clustered seqs in top 200 communities) rises from 20% to 83% — increasingly dominated by a few large communities
- **Median non-singleton size is 2 at all thresholds** — most communities are pairs, the distribution is extremely skewed

### Length vs clustering at d=10

Singleton rate is heavily driven by short contigs. Clustered count stays remarkably stable (~1.7M→1.0M) while singletons collapse with increasing min length:

| min length | sequences | singletons | % sing | clustered | % clust |
|------------|-----------|------------|--------|-----------|---------|
| 1 kbp | 6,693,829 | 5,007,254 | 74.8% | 1,686,575 | 25.2% |
| 2 kbp | 6,455,140 | 4,774,992 | 74.0% | 1,680,148 | 26.0% |
| 3 kbp | 5,951,176 | 4,286,534 | 72.0% | 1,664,642 | 28.0% |
| 4 kbp | 5,309,234 | 3,690,546 | 69.5% | 1,618,688 | 30.5% |
| 5 kbp | 4,776,770 | 3,189,618 | 66.8% | 1,587,152 | 33.2% |
| 10 kbp | 3,039,927 | 1,538,479 | 50.6% | 1,501,448 | 49.4% |
| 15 kbp | 2,117,850 | 729,068 | 34.4% | 1,388,782 | 65.6% |
| 20 kbp | 1,556,556 | 339,343 | 21.8% | 1,217,213 | 78.2% |
| 25 kbp | 1,232,447 | 188,500 | 15.3% | 1,043,947 | 84.7% |
| 30 kbp | 992,584 | 114,910 | 11.6% | 877,674 | 88.4% |
| 35 kbp | 801,315 | 72,557 | 9.1% | 728,758 | 90.9% |
| 40 kbp | 652,115 | 45,444 | 7.0% | 606,671 | 93.0% |
| 45 kbp | 542,492 | 29,646 | 5.5% | 512,846 | 94.5% |
| 50 kbp | 461,674 | 20,336 | 4.4% | 441,338 | 95.6% |

Crossover from singleton-dominated to cluster-dominated at ~10 kbp. Asymptotes to ~4-5% singletons at 50 kbp — likely genuinely novel/divergent taxa.

Largest communities are nearly unaffected by length filtering (108,799 vs 109,151 at d=10 with 10 kbp cutoff — only 0.3% loss).

### Decision: aggressive length filtering is justified

This is a question of analytical context. For a small or focused dataset, you would typically run standard 16S analysis to get community composition and identify exactly who might be present — and there, every contig matters, including short ones. But that's not our situation. We have 32 very large datasets with massive diversity, and the goal is understanding composition at scale.

**Short contigs add noise, not signal.** The k-mer profile of a 2 kbp contig is inherently noisy — there simply aren't enough bases to reliably estimate 2,772 frequency features. These contigs end up as isolated points in embedding space, inflating the singleton count without contributing to community structure.

**Community structure is robust to filtering.** The largest communities barely change when short contigs are removed (108,799 vs 109,151 at d=10 with 10 kbp cutoff — 0.3% loss). The top 10 communities are all nearly identical. The information content lives in the longer sequences.

**The clustered count is remarkably stable across cutoffs.** Going from 1 kbp to 25 kbp minimum length, the clustered count only drops from 1.69M to 1.04M, while singletons collapse from 5.0M to 189K. We're not losing community members — we're shedding unplaceable fragments.

**Filtering improves signal-to-noise in every downstream analysis.** Every plot, community characterization, and cross-dataset comparison becomes cleaner and more interpretable when the data isn't dominated by unplaceable short fragments.

**We're not losing coverage of diversity.** Organisms produce a spectrum of contig lengths during assembly. The longer contigs carry the genomic signal; the short fragments are often redundant representations of organisms already captured by their longer contigs.

Recommended cutoffs: 10 kbp (balanced — 50/50 singleton/clustered, 3.0M sequences) or 20 kbp (high confidence — 78% clustered, 1.6M sequences).

## 2026-02-16: New notebook with 10 kbp length filter

Created `clustering_010.ipynb` — copy of `clustering.ipynb` with a 10 kbp minimum contig length filter applied in Cell 1. All arrays (embeddings, kmers, embedding_ids) are filtered to >= 10 kbp (~3.04M sequences, 45.4% of original 6.7M). This notebook will be used for the length-filtered analysis going forward.

Also added sweep results plot (Cell 28) and length distribution plot (Cell 27) to `clustering.ipynb`.

### Global pairwise distance distribution confirms archipelago model (2026-02-16)

The global pairwise Euclidean distance histogram (10K sample, ~50M pairs) is tightly concentrated and unimodal: range 1.48–63.57, mean 18.70, std 3.98. On cosine distance the concentration is even more extreme (std 0.061 around mean 1.0). This is **not** a problem — it is the expected signature of an archipelago-structured latent space in 384 dimensions.

**Why the histogram looks this way:** Most random pairs come from different islands in the archipelago. In high-dimensional space, between-island distances concentrate around a common value (the "curse of dimensionality"). The histogram is dominated by these inter-island distances, which are all roughly similar. The rare close pairs (left tail) are intra-island pairs — sequences from the same dense shell.

**Why this doesn't matter for clustering:** The signal lives at the local scale, not the global scale. The neighborhood growth analysis already demonstrated this: at d=10, sequences either have ALL 50 neighbors within that radius (dense island) or NONE (isolated). The step-function behavior means distance-threshold clustering cleanly separates clusterable from unclueterable sequences without needing any global distance structure. Leiden then handles the internal community structure within the connected components.

**Key point for the paper:** The concentrated global distance distribution is evidence that the VAE embedding is working correctly — it has created a latent space where biological communities form discrete, compact islands separated by empty space. The right way to analyze this space is through local neighborhood structure (kNN graphs, distance thresholds), not global distance distributions. Standard clustering approaches that rely on global distance structure (e.g., k-means, hierarchical clustering with global linkage) would fail here, but graph-based community detection on local neighborhoods succeeds.

### 10 kbp filtered data files and kNN graph (2026-02-16)

Created filtered versions of the SFE_SE data at >= 10 kbp:
- `Runs/ids_SFE_SE_10.txt` — 3,039,927 IDs
- `Runs/embeddings_SFE_SE_10.npy` — shape (3,039,927, 385), col 0 = length
- `Runs/neighbors_SFE_SE_10.tsv` — k=50 neighbors, queried from a fresh ChromaDB loaded with only the 10 kbp+ sequences (old 6.7M ChromaDB was deleted and rebuilt)

**Important:** The neighbor relationships in `neighbors_SFE_SE_10.tsv` differ from the corresponding entries in `neighbors_SFE_SE_1.tsv` because the ChromaDB only contains 10 kbp+ sequences. Each sequence's neighbors are drawn from the filtered pool only.

### Leiden at d=10 on 10 kbp filtered data (2026-02-16)

| | Unfiltered (6.7M) | 10 kbp filtered (3.0M) |
|---|---|---|
| Singletons | 74.8% | 50.8% (1,543,718) |
| Clustered | 25.2% (1.69M) | 49.2% (1,496,209) |
| Non-singleton communities | 122K | 55,679 |
| Largest community | 109,151 | 137,065 |
| Median non-singleton size | 2 | 2 |

Confirms the length-vs-clustering prediction: singleton rate drops from 75% to 51%, clustered count barely changes (1.69M → 1.50M, only 190K lost). The largest community grew from 109K to 137K — no longer diluted by short-contig noise neighbors.

Top 5 communities: 137,065 / 55,493 / 37,054 / 28,477 / 27,502.

**This is a preliminary result at d=10.** The Leiden sweep is running overnight (`Runs/Sweep_10/`, thresholds 4.0–12.0 in 0.25 steps — finer than the original 0.5 step). Once the sweep identifies the optimal distance threshold, Leiden should be rerun at that threshold for the final community assignments. Old sweep results are in `Runs/Sweep_1/`.

**Largest community is too broad at d=10:** The 137K-sequence top community spans GC content from 23% to 67% — far too wide to be a single lineage. Contains clear same-organism triplets (e.g., SE_5/SE_10/SE_7 all at ~1,743,100 bp, 30.4% GC) but also many unrelated organisms merged via transitivity chains. Two approaches to fix:

1. **Tighter distance threshold** — reduces transitivity chain merging. The sweep will show where the largest community starts growing rapidly.
2. **Higher Leiden resolution** — splits large communities more aggressively at a given threshold. Current sweep uses resolution=1.0; a resolution sweep (e.g., 0.5, 1.0, 2.0, 5.0) should be run on top of the best distance threshold.

These interact: threshold controls which edges exist, resolution controls how aggressively Leiden partitions the graph. Need to tune both for biologically meaningful communities.

**Pairwise distance validation (d=10, unweighted Leiden):**
- Top 3 communities: mean pairwise distances 12.6–13.5 (67–72% of global mean 18.70). GC spans 30–82%. Too loose — transitivity chains merging unrelated lineages.
- Smallest non-singleton communities: tight pairs at distances 5.9–9.1, GC within 1 percentage point. These are the quality we want.

### Distance-weighted Leiden (d=10, 1/(d+0.1) weights, 2026-02-16)

Weighting edges by `1/(distance + 0.1)` had only a modest effect: largest community shrank from 137K to 119K (13% reduction), but overall structure barely changed (55,724 communities vs 55,679). Most edges are near distance 9-10 (weight ≈ 0.1), so they all look nearly identical to Leiden. The distance threshold is the dominant lever, not weighting.

### Hub discovery — root cause of over-merging (2026-02-16)

**In-degree analysis of the directed kNN graph (d<10) revealed massive hubs:**
- Top hub: 58,377 in-degree (2% of entire dataset pointing to one sequence)
- Top 20 hubs: all ~28-29% GC, 400 kbp - 1.2 Mbp (large, low-GC marine genomes)
- Distribution: median in-degree = 0 (most sequences are nobody's closest neighbor), but top hubs have tens of thousands
- P99.99 = 5,735 in-degree

These hubs are central points in embedding space where many shorter, noisier contigs from related lineages point to. They act as transitivity chain anchors, pulling thousands of loosely related sequences into the same community via A→hub→B chains.

### Mutual kNN Leiden — hub elimination (2026-02-16)

**Mutual filtering keeps only edges where both directions exist**: if B is in A's kNN AND A is in B's kNN. Hubs fail this test because they have closer neighbors than the thousands of sequences pointing at them.

Results at d=10, weighted 1/(d+0.1):

| | Symmetric weighted | Mutual weighted |
|---|---|---|
| Singletons | 50.8% | 73.9% |
| Clustered | 1,496,209 | 793,747 |
| Non-singleton communities | 55,724 | 89,168 |
| Largest | 118,858 | **5,656** |
| Mean non-singleton size | 26.9 | 8.9 |

Largest community collapsed 21× (119K → 5,656). Max mutual degree is 50 (the k=50 neighbor limit) vs 58,377 in the symmetric graph. Hubs completely eliminated.

**Pairwise distance validation (mutual, top 3):**
- Community #1 (5,656): mean dist 10.4, GC 27.0% ± 2.8%
- Community #2 (5,567): mean dist 14.2, GC 37.6% ± 2.2%
- Community #3 (4,708): mean dist 11.7, GC 28.2% ± 2.0%

GC std dropped from huge ranges (30-82%) to 2-3% — major improvement. Mean pairwise distances still above d=10 for most pairs (transitivity chains still exist, just shorter). These ~5K communities may represent genus-level groupings; higher resolution or tighter threshold could split further to species level.

**t-SNE overlay (mutual kNN Leiden, top 200 communities):** Each colored community sits in a compact, distinct region — no splattering across the map. Communities map cleanly onto the t-SNE plaque structure. Light blue (smaller non-top-200 clusters) fills many remaining dense patches. Grey singletons (74%) form the diffuse moat between islands. Top 200 cover 260,515 sequences (8.6% of total, 32.8% of clustered), smallest has 571 members. Confirms the archipelago model: discrete, compact islands with clear separation.

### In-degree capped Leiden results (d=10, cap=100, 2026-02-17)

| | Symmetric | In-degree cap=100 | Mutual |
|---|---|---|---|
| Singletons | 50.8% | 52.9% | 73.9% |
| Clustered | 1,496,209 | 1,433,109 | 793,747 |
| Non-singleton communities | 55,724 | 60,764 | 89,168 |
| Largest | 118,858 | 55,965 | 5,656 |
| Mean size | 26.9 | 23.6 | 8.9 |

Middle ground between symmetric and mutual: largest community halved (119K → 56K), only 4% coverage loss. Singleton rate barely changes (50.8% → 52.9%). Top 20 range: 55,965 down to 10,150 — still large communities, likely still have transitivity chain merging but less severe than symmetric.

**Graph saved for MCL:** `Runs/graph_capped100_d10.tsv` — ABC format (source_id\ttarget_id\tweight), 20.8M directed edges, 818 MB.

**Pairwise validation (in-degree capped top 3):**
- Community #1 (55,965): mean dist 12.5, GC 62.7% ± 3.0%, range 47–80%
- Community #2 (31,430): mean dist 12.4, GC 28.7% ± 3.5%, range 20–69%
- Community #3 (28,049): mean dist 12.3, GC 57.9% ± 3.8%, range 30–72%

At cap=100, communities are still broad (GC ranges 40–50 pp). But this is only one point in the parameter space — lower caps (e.g. 20 or 50), different distance thresholds, resolution tuning, and MCL could all produce different tradeoffs. Mutual kNN gives tighter communities (GC std 2–3%) but at the cost of coverage (26% vs 47%). The right approach depends on the use case and needs further exploration.

### Sweep_10 results (complete, 33 thresholds, 2026-02-17)

Community count peaks at d=7.0 (75,970) — earlier than unfiltered data's peak at d=8.5–9.0. Filtering short contigs tightens neighborhood structure. After d=7.0, communities merge while largest grows rapidly. Full results saved in `Runs/sweep_10_summary.tsv`. Cell 23 updated to read this file.

| d | Edges | Sing% | Communities | Largest | Top200 cov% |
|------|---------|-------|-------------|---------|------------|
| 4.0 | 911K | 92.9% | 47,129 | 1,495 | 24.7% |
| 5.0 | 2.6M | 88.3% | 62,378 | 7,232 | 33.4% |
| 6.0 | 5.8M | 82.4% | 72,230 | 16,561 | 44.5% |
| **7.0** | **11.3M** | **75.3%** | **75,970** | **31,973** | **55.7%** |
| 8.0 | 19.1M | 67.5% | 73,896 | 51,878 | 66.1% |
| 9.0 | 29.4M | 59.3% | 66,411 | 78,541 | 74.8% |
| 10.0 | 41.7M | 50.8% | 55,676 | 134,704 | 81.8% |
| 11.0 | 56.0M | 42.1% | 42,974 | 191,856 | 87.5% |
| 12.0 | 71.8M | 33.5% | 30,911 | 263,550 | 92.0% |

### In-degree capped Leiden at d=7 (cap=100, 2026-02-17)

Max in-degree at d=7: 10,304 (vs 58,377 at d=10). Cap=100 dropped 33% of edges (4.5M of 13.8M).

| | d=10 cap=100 | d=7 cap=100 | d=10 mutual |
|---|---|---|---|
| Largest | 55,965 | 13,968 | 5,656 |
| Clustered | 1,433,109 | 739,634 | 793,747 |
| Singletons | 52.9% | 75.7% | 73.9% |
| Non-singleton comm | 60,764 | 77,881 | 89,168 |
| Mean size | 23.6 | 9.5 | 8.9 |

**Pairwise validation (d=7, cap=100):**
- Community #1 (13,968): mean dist 10.1, GC 28.3% ± 1.6%, range 22–41%
- Community #2 (11,120): mean dist 9.7, GC 63.2% ± 2.5%, range 47–76%
- Community #3 (10,613): mean dist 10.3, GC 27.8% ± 3.1%, range 21–47%

**Assessment: GC ranges still too wide for biological use.** While the GC std improved (1.6–3.1% vs 3.0–3.8% at d=10), the GC ranges are still 19–26 percentage points wide. Because the VAE reconstructs GC content with very low error (~0.001 MSE on 1-mers), sequences with truly similar k-mer profiles would have near-identical GC. A 20+ pp GC range within a community means transitivity chains are still merging unrelated organisms — just shorter chains than at d=10. In-degree capping reduces the problem but does not solve it: the fundamental issue is that even without hubs, A→B→C→D chains can bridge across GC space one small step at a time. Only mutual kNN, which requires bidirectional confirmation of proximity, breaks these chains reliably.

### In-degree capped Leiden at d=5 (cap=100, 2026-02-17)

Max in-degree at d=5: 1,541 (much sparser than d=7's 10,304). Cap=100 only dropped 13% of edges (501K of 3.7M).

| | d=5 cap=100 | d=7 cap=100 | d=10 cap=100 | d=10 mutual |
|---|---|---|---|---|
| Largest | 5,620 | 13,968 | 55,965 | 5,656 |
| Clustered | 354,704 (11.7%) | 739,121 (24.3%) | 1,431,938 (47.1%) | 793,747 (26.1%) |
| Singletons | 88.3% | 75.7% | 52.9% | 73.9% |
| Non-singleton comm | 62,804 | 72,806 | 60,764 | 89,168 |
| Median size | 2 | 2 | 2 | 2 |
| Mean dist / global | 0.42–0.45 | 0.49–0.55 | — | — |
| GC range top 3 | 13, 15, 16 pp | 19, 26, 20 pp | 47, 49, 42 pp | — |

**Pairwise validation (d=5, cap=100):**
- Community #1 (5,620): mean dist 7.80, GC 28.0% ± 1.6%, range 21.5–34.6% (13 pp)
- Community #2 (2,986): mean dist 8.39, GC 28.4% ± 3.0%, range 21.2–36.2% (15 pp)
- Community #3 (2,738): mean dist 7.89, GC 43.2% ± 3.1%, range 34.2–50.6% (16 pp)

**Trend:** GC ranges shrink with lower d (13–16 pp at d=5 vs 19–26 at d=7 vs 40–50 at d=10), but are still too wide for biological clustering. 13 pp GC spread means transitivity chains persist even at this tight threshold. Coverage drops sharply — only 11.7% clustered.

### MCL sweep results (2026-02-17)

MCL inflation sweep on `Runs/graph_capped100_d10.tsv` (d=10 capped graph, 20.8M edges).
Script in `Runs/MCL/run_mcl_sweep.sh`. Binary graph cached in `Runs/MCL/graph.mci` + `graph.tab`.

**Important caveat:** MCL only sees the 1,433,109 nodes present in the graph (47.1% of 3,039,927 total). The remaining 1,606,818 isolated sequences are not in the ABC file.

| I | Non-sing clusters | Largest | Clustered (of total) | MCL singletons |
|---|---|---|---|---|
| 1.4 | 109,579 | 3,208 | 1,433,109 (47.1%) | 0 |
| 2.0 | 174,972 | 431 | 1,430,501 (47.1%) | 2,608 |
| 3.0 | 238,597 | 216 | 1,408,912 (46.3%) | 24,197 |
| 4.0 | 273,050 | 204 | 1,376,276 (45.3%) | 56,833 |
| 5.0 | 292,252 | 180 | 1,345,284 (44.3%) | 87,825 |
| 6.0 | 303,679 | 142 | 1,319,185 (43.4%) | 113,924 |

**GC validation (top 3 communities per inflation):**

| I | GC spans (pp) | GC std (%) | Sizes |
|---|---|---|---|
| 1.4 | 18, 21, 16 | 1.9, 3.9, 2.1 | 3208, 2198, 2001 |
| 2.0 | 10, 10, 16 | 1.4, 0.8, 1.4 | 431, 420, 361 |
| 3.0 | 9, 7, 7 | 1.3, 1.3, 1.3 | 216, 189, 186 |
| 4.0 | 9, 7, 13 | 1.3, 1.3, 2.2 | 204, 174, 164 |
| 5.0 | 9, 7, 6 | 1.3, 1.3, 1.0 | 180, 163, 155 |
| 6.0 | 6, 7, 13 | 1.0, 1.1, 2.4 | 142, 141, 133 |

**Key finding: MCL breaks transitivity chains.** At I≥3.0, GC spans drop to 6–9 pp with std ~1.0–1.3% — dramatically better than any Leiden configuration tested. Compare:
- Leiden capped d=10: 42–49 pp
- Leiden capped d=7: 19–26 pp
- Leiden capped d=5: 13–16 pp
- **MCL I=3–5 on d=10 graph: 6–9 pp**

MCL's flow-based approach naturally attenuates weak transitive connections through iterative expansion and inflation, whereas Leiden's modularity optimization merges anything reachable. The trade-off is cluster size — MCL largest is 142–216 at I≥3.0 vs Leiden's thousands. But for biological coherence these are far superior.

The I=1.4 result (18–21 pp) still shows chain contamination — the inflation is too weak to break flows. The sweet spot appears to be I=3.0–5.0: tight GC, reasonable cluster counts (238K–292K non-singleton), and nearly all connected nodes clustered (93–98% of in-graph nodes).

### MCL d=5 sweep results (2026-02-17)

MCL on `Runs/graph_capped100_d5.tsv` (d=5 capped graph, 354,704 nodes, 4.2M edges). Binary graph + cluster files in `Runs/MCL_d5/`. MCL sweep script moved to `Runs/run_mcl_sweep.sh`, MCL d=10 results moved to `Runs/MCL_d10/`.

Jury pruning scores: "superb" (I=1.4), "marvelous" (I=2–3), "perfect" (I=4–6).

| I | Non-sing clusters | Largest | Clustered (of total) | MCL singletons |
|---|---|---|---|---|
| 1.4 | 72,284 | 1,055 | 354,704 (11.7%) | 0 |
| 2.0 | 82,052 | 281 | 354,676 (11.7%) | 28 |
| 3.0 | 90,323 | 169 | 353,615 (11.6%) | 1,089 |
| 4.0 | 95,654 | 153 | 351,168 (11.6%) | 3,536 |
| 5.0 | 99,077 | 138 | 348,755 (11.5%) | 5,949 |
| 6.0 | 101,460 | 128 | 346,717 (11.4%) | 7,987 |

**GC validation (top 3 communities per inflation):**

| I | GC spans (pp) | GC std (%) | Sizes |
|---|---|---|---|
| 1.4 | 9, 8, 8 | 0.9, 0.8, 1.0 | 1055, 979, 889 |
| 2.0 | 5, 5, 5 | 0.9, 0.7, 0.8 | 281, 264, 263 |
| 3.0 | 6, 4, 8 | 1.2, 0.6, 1.8 | 169, 166, 160 |
| 4.0 | 6, 4, 6 | 1.1, 0.7, 0.9 | 153, 134, 132 |
| 5.0 | 6, 8, 4 | 1.1, 0.9, 0.7 | 138, 128, 127 |
| 6.0 | 8, 6, 5 | 0.9, 1.1, 0.8 | 128, 122, 122 |

**Cross-graph MCL comparison (GC spans in top 3, pp):**

| I | d=10 MCL | d=5 MCL |
|---|---|---|
| 1.4 | 18, 21, 16 | 9, 8, 8 |
| 2.0 | 10, 10, 16 | **5, 5, 5** |
| 3.0 | 9, 7, 7 | 6, 4, 8 |
| 4.0 | 9, 7, 13 | 6, 4, 6 |
| 5.0 | 9, 7, 6 | 6, 8, 4 |
| 6.0 | 6, 7, 13 | 8, 6, 5 |

**Key observations:**
- d=5 MCL at I=2.0 achieves the best GC coherence seen: **5 pp span, 0.7–0.9% std** across all top 3. Mean pairwise distances 5.1–5.8, well below the d=5 threshold.
- At low inflation (I=1.4), d=5 already outperforms d=10 at I=3.0 — the tighter graph gives MCL less chain material to work with.
- At higher inflation (I≥3.0), d=5 and d=10 converge toward similar GC spans (4–9 pp), suggesting the inflation is doing most of the work at that point.
- Coverage trade-off: d=5 clusters only 11.4–11.7% of sequences vs 43–47% for d=10. The 88% singletons are sequences with no neighbors within d=5.

### Weight transformation and mutual kNN experiments (2026-02-17)

Tested two additional MCL configurations to isolate the effects of weight function and graph construction:

1. **d=10 capped, exp(-d) weights** → `Runs/MCL_d10_expd/`
2. **Mutual kNN (all k=50 neighbors, no d threshold), exp(-d) weights** → `Runs/MCL_mutual_expd/`

Graphs exported from notebook cells 34–35. MCL sweep script at `Runs/run_mcl_sweep.sh`.

**Mutual kNN exp(-d):** 925,144 nodes (30.4% of total), nearly zero MCL singletons.
**d=10 capped exp(-d):** 1,433,109 nodes (47.1%), same graph topology as before but different weights.

**Grand comparison — GC spans (pp) in top 3 communities:**

| Method | I=1.4 | I=2.0 | I=3.0 | I=5.0 | Coverage |
|---|---|---|---|---|---|
| d=10 capped 1/(d+0.1) | 18, 21, 16 | 10, 10, 16 | 9, 7, 7 | 9, 7, 6 | 47% |
| **d=10 capped exp(-d)** | 47, 50, 35 | 50, 46, 20 | 20, 18, 45 | 20, 18, 9 | 47% |
| d=5 capped 1/(d+0.1) | 9, 8, 8 | **5, 5, 5** | 6, 4, 8 | 6, 8, 4 | 12% |
| **Mutual kNN exp(-d)** | 9, 6, 12 | 8, 27, 9 | 8, 6, 7 | 8, 4, 8 | 30% |

**Key findings:**

- **exp(-d) on d=10 capped is much worse** than 1/(d+0.1). GC spans of 18–50 pp at all inflations. The exponential weighting creates strong local cores, but the larger clusters (4K–7K at I=1.4–2.0) absorb chains through concentrated flow around hubs. The sharp weight contrast amplifies rather than suppresses the chain problem on dense graphs.

- **Mutual kNN exp(-d) is decent but inconsistent.** Most top-3 communities have 6–9 pp GC spans, but outliers appear (27 pp at I=2.0 #2, 15 pp at I=6.0 #3). The chain-breaking from mutual filtering helps, but long-distance mutual pairs (d=10–15, mean pairwise distances often 8–14) still contaminate some clusters. Coverage is 30.4% — better than d=5 (12%) but quality is not as reliable.

- **1/(d+0.1) outperforms exp(-d) on the same graph.** The gentler similarity function works better for MCL because it maintains more uniform flow across moderate-distance edges, letting inflation do the separation work rather than pre-concentrating flow on close pairs.

### MCL d=7 results (2026-02-17)

MCL on `Runs/graph_capped100_d7.tsv` (d=7 capped, 1/(d+0.1) weights, 9.3M edges, 739,634 nodes = 24.3% of total). Results in `Runs/MCL_d7/`.

| I | Non-sing clusters | Largest | Clustered (of total) | MCL singletons |
|---|---|---|---|---|
| 1.4 | 104,941 | 2,326 | 739,634 (24.3%) | 0 |
| 2.0 | 131,996 | 369 | 739,176 (24.3%) | 458 |
| 3.0 | 157,407 | 177 | 732,642 (24.1%) | 6,992 |
| 4.0 | 172,450 | 144 | 721,461 (23.7%) | 18,173 |
| 5.0 | 181,224 | 142 | 711,315 (23.4%) | 28,319 |
| 6.0 | 186,921 | 134 | 702,756 (23.1%) | 36,878 |

**Complete MCL comparison — GC spans (pp) in top 3, 1/(d+0.1) weights:**

| I | d=5 (12%) | d=7 (24%) | d=10 (47%) |
|---|---|---|---|
| 1.4 | 9, 8, 8 | 24, 13, 11 | 18, 21, 16 |
| 2.0 | **5, 5, 5** | 10, 10, 8 | 10, 10, 16 |
| 3.0 | 6, 4, 8 | **6, 4, 7** | 9, 7, 7 |
| 4.0 | 6, 4, 6 | 8, 7, 9 | 9, 7, 13 |
| 5.0 | 6, 8, 4 | 8, 8, 9 | 9, 7, 6 |
| 6.0 | 8, 6, 5 | **6, 6, 9** | 6, 7, 13 |

**Key findings:**
- **d=7 at I=3.0 is the sweet spot for coverage/coherence**: 6, 4, 7 pp GC spans with 24% coverage — double the coverage of d=5 with comparable quality. Largest cluster 177.
- At I=2.0, d=7 still shows some chain contamination (10, 10, 8 pp) while d=5 is clean (5, 5, 5). But at I≥3.0, d=7 converges to the same quality as d=5.
- At I=1.4, d=7 is worst (24 pp) — the graph is dense enough for chains at low inflation.
- The pattern is clear: **graph sparsity and inflation are partially interchangeable** — a sparser graph (d=5) needs less inflation (I=2.0), while a denser graph (d=7, d=10) needs I≥3.0 to achieve the same GC coherence.

**Overall MCL summary (1/(d+0.1) weights):**
- Best coherence: d=5, I=2.0 — 5 pp, 12% coverage
- Best coherence/coverage balance: d=7, I=3.0 — 6, 4, 7 pp, 24% coverage
- Highest coverage with acceptable quality: d=10, I=3.0 — 9, 7, 7 pp, 47% coverage

### Remaining action items

- **Size distribution analysis** — characterize full MCL cluster size distribution beyond top 3
- **Cross-method validation** — clusters stable across both Leiden and MCL are high-confidence
- **Update sweep plot** (Cell 23) with new `sweep_10_summary.tsv`
- **Commit and push** — large batch of new results
