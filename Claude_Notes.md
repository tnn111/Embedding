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

**Length sweep results (2026-02-08):**
- All 5 runs completed (1,000 epochs each), verify_local_distances run on all
- Best Spearman: Run 1 (1,000 bp) at 0.852 — Jeffreys prior solved the short-contig problem
- Results not monotonic: Run 4 (4,000 bp) worst at 0.650
- All runs below the 0.93 Spearman of the previous single-run model (different data mix)
- Cross-comparison on common 5k data: Run 5 best (0.731), Runs 1-3 close (0.694-0.717), Run 4 outlier (0.580)
- **Run 4 outlier root cause: ReduceLROnPlateau scheduling artifact**
  - All other runs: 1st LR reduction at epoch 21-23 (right after KL warmup plateau)
  - Run 4: 1st LR reduction at epoch **566** — val_loss on a long slow descent, never triggered patience=20
  - Run 4 spent only 146 epochs at min LR vs 547-661 for others
  - Run 4 has the best reconstruction MSE (especially 5-mer, 3-mer, 2-mer) but worst Spearman
  - Classic reconstruction-representation tradeoff: high LR prevented latent space local structure from stabilizing
  - NOT a convergence issue — the loss converged fine, but the latent space organization suffered
  - **Fix: retraining as Run_4_prime with starting LR=5e-5** — started 2026-02-08
  - Run_4_prime LR schedule is normal: reductions at epochs 266, 292, 365 (vs epoch 566 for original Run 4)
  - Full LR schedule normal: reductions at 266, 292, 365, 388, 409
  - Min LR at epoch 459 (541 epochs at min LR — comparable to other runs)
  - **Final: Spearman 0.727 on common 5k data, 0.786 on own 4k data**
  - Falls between Run 3 (0.717) and Run 5 (0.731) — exactly where 4,000 bp threshold should be
  - With Run 4' corrected, Spearman increases monotonically with threshold: 0.694→0.712→0.717→0.727→0.731
  - Confirms original outlier was entirely a scheduling artifact, not a threshold effect
- FD paper confirms 3 kbp minimum is standard (mmlong2 pipeline default)
- Full results in VAE.md under "2026-02-08: Minimum contig length sweep results"

**Run_SFE_SE (2026-02-08):**
- Aquatic-only (SFE+SE ≥ 5kbp), 4.8M samples, new preprocessing (per-group CLR + Jeffreys)
- Final: Val=184.8, MSE=0.055, KL=403 (lowest of any run)
- Short k-mers dramatically better than sweep runs (4-mer 0.0013 vs 0.0024), 6-mer slightly worse
- Spearman 0.714 on common 5k test data — on par with Runs 1-3 despite 3x fewer samples
- LR schedule: 1st reduction at epoch 248 (between sweep runs and Run 4)
- Key insight: diversity (FD + RefSeq) helps reconstruction more than retrieval quality

**Key insight: reconstruction loss is not enough**
- Run 4 proved that best reconstruction MSE ≠ best embedding quality
- Need independent metrics: Spearman (ranking) and count-vs-distance linearity (geometry)
- The LR scheduling artifact would not have been visible at typical training lengths (100-500 epochs)
- Worth highlighting in the paper: most metagenomic embedding papers only report reconstruction loss
- TODO: re-run count-vs-distance analysis on final model to confirm linear regime persists

**Full cross-comparison matrix (2026-02-08):**
Tested all 6 models on test data at each threshold (1k, 2k, 3k, 5k). Spearman:

| Model | 1k data | 2k data | 3k data | 4k data | 5k data |
|-------|---------|---------|---------|---------|---------|
| Run 1 (1k) | 0.852 | 0.720 | 0.680 | 0.755 | 0.694 |
| Run 2 (2k) | 0.867 | 0.742 | 0.710 | 0.786 | 0.712 |
| Run 3 (3k) | **0.871** | 0.730 | **0.714** | **0.797** | 0.717 |
| Run 4' (4k) | 0.869 | 0.734 | 0.707 | 0.786 | 0.727 |
| Run 5 (5k) | 0.847 | **0.742** | 0.657 | 0.749 | **0.731** |
| SFE_SE_5 | 0.851 | 0.696 | 0.686 | 0.760 | 0.714 |

Key findings:
- No single model dominates all test conditions
- Run 3 (3k) is the best generalist — highest on 1k, 3k, and 4k, competitive elsewhere
- Shorter test data is "easier" — all models 0.847-0.871 on 1k vs 0.657-0.731 on 5k
- The monotonic trend on 5k data doesn't hold for other test sets
- Run 5 is polarized: best on 5k but worst on 3k (0.657) and near-worst on 4k (0.749)
- Models aren't best on their "own" threshold data — Run 3 beats Run 4' on 4k data

**SFE_SE cross-comparison (2026-02-09):**
Trained Run_SFE_SE_1 through _4, tested all 6 sweep models on all 5 SFE_SE test datasets:

| Model | SFE_SE_1 | SFE_SE_2 | SFE_SE_3 | SFE_SE_4 | SFE_SE_5 |
|-------|----------|----------|----------|----------|----------|
| Run 1 (1k) | 0.761 | 0.747 | 0.831 | 0.871 | 0.782 |
| Run 2 (2k) | 0.791 | 0.764 | 0.844 | 0.874 | 0.778 |
| Run 3 (3k) | 0.779 | 0.773 | 0.841 | 0.859 | 0.761 |
| Run 4' (4k) | 0.789 | 0.779 | 0.808 | 0.841 | 0.754 |
| Run 5 (5k) | 0.790 | 0.763 | 0.700 | 0.785 | 0.695 |
| SFE_SE_5 | 0.777 | 0.776 | 0.744 | 0.842 | 0.742 |

Key: Run 2 best on aquatic data (not Run 3 as on mixed). Run 5 collapses on SFE_SE_3/5. Sweep models beat aquatic-only model on its own data. Reversed monotonic trend on SFE_SE_5 test.

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

### 2026-02-10/11/12: Shuffled retraining sweep — correlation analysis

Rerunning all models on shuffled data (1000 epochs each). Correlation results so far (verify_local_distances, 50k samples, own data):

| Run | Epochs | Spearman | Pearson | Top 1 MSE | Random MSE |
|-----|--------|----------|---------|-----------|------------|
| Run 1 (1k) | 1000 | 0.768 | 0.525 | 0.109 | 0.456 |
| Run 2 (2k) | 1000 | 0.627 | 0.360 | 0.167 | 0.542 |
| Run 3 (3k) | 1000 | 0.721 | 0.388 | 0.119 | 0.511 |
| Run 4 (4k) | 1000 | 0.782 | 0.528 | 0.121 | 0.516 |
| Run 5 (5k) | ~378 | 0.690 | 0.402 | 0.109 | 0.492 |

**Key findings:**
- Spearman values are lower than unshuffled runs (e.g., Run 1: 0.852→0.768) because the old unshuffled validation set was biased toward one source, making the ranking task artificially easier
- Run 4 has the best Spearman (0.782) despite having started with LR=1e-5
- Run 2 is an unexplained outlier at 0.627
- Convergence-by-500 pattern confirmed: Run 1 (0.766→0.768 from epoch 570→750), Run 3 (0.724→0.721 from epoch 480→1000), Run 4 (0.789→0.782 from epoch 184→1000)
- 500 epochs appears sufficient; remaining epochs provide negligible improvement

**Train/val gap analysis — BatchNorm artifact:**

| Run | Train | Val | Gap | Sequences |
|-----|-------|-----|-----|-----------|
| Run 1 (1k) | 189.3 | 189.2 | 0.1 | 17.6M |
| Run 2 (2k) | 178.1 | 177.6 | 0.5 | 17.1M |
| Run 3 (3k) | 166.3 | 165.9 | 0.3 | 16.5M |
| Run 4 (4k) | 142.3 | 190.4 | 48.1 | 14.8M |
| Run 5 (5k) | 120.8 | 162.6 | 41.8 | 13.4M |

The gap in Runs 4-5 is NOT overfitting. Evidence:
- Gap exists from epoch 1 (Run 5: Train 156.6, Val 179.1 before any learning)
- Old and new Run_5 reach identical val loss (162.4 vs 162.6) — same generalization
- Root cause: BatchNorm uses per-batch statistics during training (training=True) but running statistics during validation (training=False)
- The old "Recon" metric was computed with training=False on validation data, so it could never detect this gap — old Recon+beta*KL ≈ Val because both were validation-mode measurements
- The metrics fix (replacing Recon with actual Keras training loss) made the BN gap visible
- BN effect is larger for 4k/5k data (more distinctive, varied k-mer profiles) than 1k data (noisier short sequences dampen batch-to-batch variation)
- The magnitude difference (0.1 for Run 1 vs 42 for Run 5) deserves further investigation

**Remaining runs:** Run 5 finishing, then SFE_SE_1 through SFE_SE_5. ~3-4 days total GPU time. Full cross-comparison and final conclusions after all complete.

**ClusteringPaper repo updated** to commit 97a70ac (pulled 2026-02-12).
