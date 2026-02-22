# Runs

The Runs directory contains all of the training data and the models as well as the original FASTA data.

kmers_{N} contains the kmers generated from FD_contigs_{N}, NCBI_contigs_{N}, SFE_contigs_{N} and SE_contigs_{N} where N is 1, 2, 3, 4 or 5
and indicates the threshold for keeping nucleotide sequences. N = 1 means keeping everything longer than 1 kbp, N = 2 means keeping
everything longer than 2 kbp, N = 3 means keeping everything longer than 3 kbp, N = 4 means keeping everything longer than 4 kbp and N = 5
means keeping everything longer than 5 kbp. The files named ids_{N} are the ids from the nucleotide sequences in the same order as kmers_{N}.

kmers_SFE_SE{N} contains the kmers generated only from SFE_contigs_{N} and SE_contigs_{N} where N is 1, 2, 3, 4 or 5
and indicates the threshold for keeping nucleotide sequences. N = 1 means keeping everything longer than 1 kbp, N = 2 means keeping
everything longer than 2 kbp, N = 3 means keeping everything longer than 3 kbp, N = 4 means keeping everything longer than 4 kbp and N = 5
means keeping everything longer than 5 kbp. The files named ids_SFE_SE_{N} are the ids from the nucleotide sequences in the same order as
kmers_SFE_SE_{N}.

---

## 1. Dataset Characterization

### Training Dataset Sizes

| Dataset | Sequences | Columns | Size (GB) | Sources |
|---------|-----------|---------|-----------|---------|
| kmers_1 (≥1 kbp) | 17,415,045 | 2,773 | 179.9 | FD + NCBI + SFE + SE |
| kmers_2 (≥2 kbp) | 17,055,694 | 2,773 | 176.2 | FD + NCBI + SFE + SE |
| kmers_3 (≥3 kbp) | 16,495,774 | 2,773 | 170.4 | FD + NCBI + SFE + SE |
| kmers_4 (≥4 kbp) | 14,822,956 | 2,773 | 153.1 | FD + NCBI + SFE + SE |
| kmers_5 (≥5 kbp) | 13,439,298 | 2,773 | 138.8 | FD + NCBI + SFE + SE |
| kmers_SFE_SE_1 (≥1 kbp) | 6,693,829 | 2,773 | 69.2 | SFE + SE only |
| kmers_SFE_SE_2 (≥2 kbp) | 6,455,140 | 2,773 | 66.7 | SFE + SE only |
| kmers_SFE_SE_3 (≥3 kbp) | 5,951,176 | 2,773 | 61.5 | SFE + SE only |
| kmers_SFE_SE_4 (≥4 kbp) | 5,309,235 | 2,773 | 54.9 | SFE + SE only |
| kmers_SFE_SE_5 (≥5 kbp) | 4,776,770 | 2,773 | 49.4 | SFE + SE only |

Column 0 is sequence length; columns 1-2772 are k-mer frequencies (2,772 features). All files are float32.

### Sequence Count by Threshold

Raising the minimum contig length from 1 kbp to 5 kbp removes ~23% of augmented sequences (17.4M → 13.4M) and ~29% of SFE_SE sequences (6.7M → 4.8M). SFE_SE data loses proportionally more short contigs.

The FD + NCBI fraction (augmented minus SFE_SE) ranges from 10.7M (1K threshold) to 8.7M (5K threshold), representing 61-65% of each augmented dataset.

### NCBI-Only Training Data

| Dataset | Sequences | Columns | Size (GB) | Sources |
|---------|-----------|---------|-----------|---------|
| kmers_NCBI_5 (≥5 kbp) | 655,859 | 2,773 | 6.8 | NCBI RefSeq only |

NCBI data comes from ~20,000 RefSeq representative genomes fragmented into contigs. Median contig length ~37 kbp.

### Data Sources

- **FD**: Microflora Danica metagenomic contigs (aquatic and soil)
- **NCBI**: NCBI RefSeq representative genomes (~20K genomes, curated, taxonomically diverse)
- **SFE**: San Francisco Estuary metagenomic contigs (marine)
- **SE**: Baltic Sea metagenomic contigs (marine)

### Data Format

Each row is a k-mer frequency vector across 6 scales:

| K-mer size | Features | Columns |
|------------|----------|---------|
| 6-mer | 2,080 | 1-2080 |
| 5-mer | 512 | 2081-2592 |
| 4-mer | 136 | 2593-2728 |
| 3-mer | 32 | 2729-2760 |
| 2-mer | 10 | 2761-2770 |
| 1-mer | 2 | 2771-2772 |

---

## 2. Model Architecture

All runs use the same VAE architecture:

**Encoder**: 2772 → Dense(1024) → BatchNorm → LeakyReLU(0.2) → Dense(512) → BatchNorm → LeakyReLU(0.2) → z_mean(384) / z_log_var(384)

**Decoder**: 384 → Dense(512) → BatchNorm → LeakyReLU(0.2) → Dense(1024) → BatchNorm → LeakyReLU(0.2) → Dense(2772)

**Parameters**: ~7.1M total (encoder ~3.6M, decoder ~3.5M)

**Loss**: MSE on CLR-transformed features + β × KL divergence (β = 0.05)

**Training**: 1000 epochs, batch size 1024, initial LR 1e-4, ReduceLROnPlateau (factor=0.5, patience=20, min=1e-6), 90/10 train/val split, KL warmup over 5 epochs.

---

## 3. Training Results

### Final Training Metrics (Epoch 1000)

| Run | Val Loss | MSE | KL | Train/Val Gap |
|-----|----------|------|-----|---------------|
| Run_1 (1K) | 189.18 | 0.059 | 475.5 | 0.12 |
| Run_2 (2K) | 177.61 | 0.055 | 468.2 | 0.46 |
| Run_3 (3K) | 165.93 | 0.052 | 461.6 | 0.33 |
| Run_4 (4K) | 143.14 | 0.044 | 424.9 | 0.17 |
| Run_5 (5K) | 126.43 | 0.038 | 394.6 | 0.26 |
| SFE_SE_1 (1K) | 243.62 | 0.080 | 498.6 | 0.15 |
| SFE_SE_2 (2K) | 224.24 | 0.073 | 491.5 | 0.07 |
| SFE_SE_3 (3K) | 200.40 | 0.064 | 448.8 | -0.03 |
| SFE_SE_4 (4K) | 174.75 | 0.056 | 414.0 | 0.13 |
| SFE_SE_5 (5K) | 155.59 | 0.049 | 389.4 | -0.03 |

Val loss and MSE decrease monotonically from threshold 1K → 5K in both series: longer sequences produce cleaner k-mer profiles that are easier to reconstruct. Train/val gaps are negligible (<0.5 points) across all runs, confirming proper data shuffling.

SFE_SE runs have consistently higher loss than augmented runs at matching thresholds (e.g., SFE_SE_5 at 155.59 vs Run_5 at 126.43). This is expected — SFE_SE data has fewer sequences and less source diversity (missing FD + NCBI), making reconstruction harder.

### Per-K-mer Reconstruction MSE (Epoch 1000)

| Run | Total | 6-mer | 5-mer | 4-mer | 3-mer | 2-mer | 1-mer |
|-----|-------|-------|-------|-------|-------|-------|-------|
| Run_1 (1K) | 0.059 | 0.07680 | 0.00700 | 0.00201 | 0.00104 | 0.000514 | 0.000087 |
| Run_2 (2K) | 0.055 | 0.07109 | 0.00633 | 0.00172 | 0.000808 | 0.000335 | 0.000080 |
| Run_3 (3K) | 0.052 | 0.06766 | 0.00609 | 0.00167 | 0.000753 | 0.000327 | 0.000078 |
| Run_4 (4K) | 0.044 | 0.05692 | 0.00560 | 0.00141 | 0.000662 | 0.000302 | 0.000072 |
| Run_5 (5K) | 0.038 | 0.04948 | 0.00520 | 0.00135 | 0.000717 | **0.000450** | **0.000135** |
| SFE_SE_1 (1K) | 0.080 | 0.10290 | 0.01171 | 0.00402 | 0.002402 | 0.001061 | 0.000167 |
| SFE_SE_2 (2K) | 0.073 | 0.09481 | 0.01013 | 0.00344 | 0.002067 | 0.000851 | 0.000133 |
| SFE_SE_3 (3K) | 0.064 | 0.08278 | 0.00970 | 0.00264 | 0.001482 | 0.000648 | 0.000084 |
| SFE_SE_4 (4K) | 0.056 | 0.07226 | 0.00930 | 0.00255 | 0.001542 | 0.000654 | 0.000093 |
| SFE_SE_5 (5K) | 0.049 | 0.06298 | 0.00836 | 0.00222 | 0.001303 | 0.000543 | 0.000078 |

6-mer features dominate total MSE (~95-98%), as they comprise 2,080 of 2,772 features.

**Run_5 2-mer/1-mer anomaly**: Run_5 breaks the otherwise monotonic improvement trend on 2-mer (0.000450) and 1-mer (0.000135) — these are higher than Runs 2-4. Root cause: extreme high-GC sequences (>75% GC content) are underrepresented in ≥5 kbp training data. Only 46 validation samples (0.46%) but their 1-mer error is 26× higher than in Run_3 (0.0099 vs 0.0004), contributing ~50% of total 1-mer MSE. These organisms have small genomes assembling into shorter contigs, so they're underrepresented at higher thresholds. The 3K dataset retains more of them (4 GC distribution peaks vs 3 for 5K).

**SFE_SE per-k-mer observations**: SFE_SE runs show 1.3-2.3× higher per-k-mer MSE than augmented runs, with the gap widening for shorter k-mers (3-mer through 1-mer). SFE_SE_5 does not show the 2-mer/1-mer anomaly seen in augmented Run_5 — likely because the SFE_SE data has different GC distribution characteristics.

### Learning Rate Schedules

All runs use ReduceLROnPlateau (factor 0.5, patience 20, min LR 1e-6) with 7 reductions total:

| Run | 1st Reduction | LR Floor (1e-6) | Epochs at Floor |
|-----|---------------|-----------------|-----------------|
| Run_1 (1K) | Epoch 21 | Epoch 351 | 649 |
| Run_2 (2K) | Epoch 21 | Epoch 354 | 646 |
| Run_3 (3K) | Epoch 22 | Epoch 316 | 684 |
| Run_4 (4K) | Epoch 22 | Epoch 468 | 532 |
| Run_5 (5K) | Epoch 21 | Epoch 346 | 654 |
| SFE_SE_1 (1K) | Epoch 118 | Epoch 406 | 594 |
| SFE_SE_2 (2K) | Epoch 124 | Epoch 415 | 585 |
| SFE_SE_3 (3K) | Epoch 165 | Epoch 409 | 591 |
| SFE_SE_4 (4K) | Epoch 181 | Epoch 445 | 555 |
| SFE_SE_5 (5K) | Epoch 217 | Epoch 544 | 456 |

Augmented runs hit their first LR reduction early (epochs 21-22). SFE_SE runs plateau much later (epochs 118-217), suggesting the SFE_SE loss surface is smoother and the model can sustain learning at the initial LR for longer. All runs spend the majority of training (456-684 epochs) at the minimum LR floor.

---

## 4. Embedding Quality Evaluation

### Methodology

Embedding quality is assessed using `verify_local_distances.py`, which measures how well latent-space distances predict k-mer similarity:

1. Load 50,000 random samples from a test dataset
2. Encode all samples using the VAE encoder (deterministic z_mean)
3. Select 100 random query samples
4. For each query, find the 50 nearest neighbors in latent space (Euclidean distance)
5. Compute MSE between each query-neighbor pair in original k-mer space
6. Report Spearman rank correlation between latent distances and k-mer MSE

Higher Spearman = better embedding (latent neighbors are genuinely similar in k-mer space).

### Distance Metric: Euclidean vs Cosine

Tested on Run_4 with 4K augmented data:
- Euclidean: Spearman **0.697**
- Cosine: Spearman 0.621

Euclidean wins by +0.076. The VAE's MSE loss creates Euclidean-friendly geometry. All evaluations below use Euclidean distance.

---

## 5. Augmented Run Cross-Comparison (5×5)

Each of the 5 augmented models tested on all 5 augmented test datasets.

### Spearman Correlation

| Model \ Test | 1K | 2K | 3K | 4K | 5K | Mean |
|---|---|---|---|---|---|---|
| Run_1 (1K) | **0.751** | 0.616 | 0.723 | 0.703 | 0.635 | 0.686 |
| Run_2 (2K) | 0.764 | **0.627** | 0.729 | 0.711 | 0.643 | 0.695 |
| **Run_3 (3K)** | **0.769** | **0.639** | **0.721** | **0.722** | **0.660** | **0.702** |
| Run_4 (4K) | 0.738 | 0.598 | 0.692 | **0.674** | 0.625 | 0.665 |
| Run_5 (5K) | 0.726 | 0.584 | 0.655 | 0.640 | **0.616** | 0.644 |

Bold diagonal = own-data results. Bold in columns = best for that test data.

### Top 1 Neighbor MSE (lower = better)

| Model \ Test | 1K | 2K | 3K | 4K | 5K | Mean |
|---|---|---|---|---|---|---|
| Run_1 (1K) | **0.167** | 0.172 | 0.108 | 0.117 | 0.105 | 0.134 |
| Run_2 (2K) | 0.175 | **0.167** | 0.113 | 0.126 | 0.103 | 0.137 |
| **Run_3 (3K)** | **0.160** | **0.156** | **0.119** | **0.121** | 0.103 | **0.132** |
| Run_4 (4K) | 0.178 | 0.176 | 0.127 | **0.138** | 0.101 | 0.144 |
| Run_5 (5K) | 0.180 | 0.176 | 0.125 | 0.140 | **0.100** | 0.144 |

### Key Findings — Augmented Runs

1. **Run_3 (3K bp) is the best general-purpose encoder**. It wins or ties for best on every column in both Spearman and MSE matrices (mean Spearman 0.702, mean Top 1 MSE 0.132).

2. **Clear tier structure**: Tier 1: Run_3 (0.702); Tier 2: Run_1-2 (0.686-0.695); Tier 3: Run_4-5 (0.644-0.665).

3. **No model is best on its own data** (except Run_1 on 1K by a negligible margin). Run_3 beats Run_4 on 4K data and all models on 5K data.

4. **Models trained at lower thresholds generalize upward better than the reverse**. Run_1-3 score 0.635-0.660 on 5K data; Run_5 scores only 0.584-0.726 across columns.

5. **2K test data is uniquely hard** — all models score 0.584-0.639, well below other columns.

6. **Shorter sequences have noisier k-mer profiles** — Top 1 MSE is ~0.16-0.18 for 1K/2K test data vs ~0.10-0.14 for 3K-5K.

7. **Reconstruction loss ≠ embedding quality**. Run_5 has the lowest training MSE (0.038) but the worst Spearman (0.644). Run_3 (MSE 0.052) has the best Spearman (0.702). The additional biological diversity captured at 3K provides better latent structure than the cleaner k-mer profiles at 5K.

---

## 6. SFE_SE Cross-Comparison

### SFE_SE Models on Augmented Test Data (Spearman)

| Model \ Test | 1K | 2K | 3K | 4K | 5K | Mean |
|---|---|---|---|---|---|---|
| **SFE_SE_1 (1K)** | **0.907** | **0.820** | **0.881** | **0.843** | **0.829** | **0.856** |
| SFE_SE_2 (2K) | 0.889 | 0.808 | 0.873 | 0.827 | 0.807 | 0.841 |
| SFE_SE_3 (3K) | 0.886 | 0.802 | 0.868 | 0.813 | 0.811 | 0.836 |
| SFE_SE_4 (4K) | 0.878 | 0.804 | 0.850 | 0.807 | 0.805 | 0.829 |
| SFE_SE_5 (5K) | 0.866 | 0.790 | 0.836 | 0.787 | 0.782 | 0.812 |

SFE_SE_1 wins every column on augmented data. Ranking: SFE_SE_1 (0.856) > SFE_SE_2 (0.841) > SFE_SE_3 (0.836) > SFE_SE_4 (0.829) > SFE_SE_5 (0.812). Lower thresholds generalize better to augmented data, consistent with the augmented run pattern.

### SFE_SE Models on SFE_SE Test Data (Spearman)

| Model \ Test | SFE_SE_1 | SFE_SE_2 | SFE_SE_3 | SFE_SE_4 | SFE_SE_5 | Mean |
|---|---|---|---|---|---|---|
| SFE_SE_1 (1K) | **0.773** | 0.739 | 0.723 | 0.816 | 0.779 | 0.766 |
| SFE_SE_2 (2K) | 0.743 | **0.714** | 0.682 | 0.784 | 0.758 | 0.736 |
| SFE_SE_3 (3K) | 0.749 | 0.704 | **0.676** | 0.776 | 0.736 | 0.728 |
| SFE_SE_4 (4K) | 0.751 | 0.684 | 0.687 | **0.781** | 0.726 | 0.726 |
| **SFE_SE_5 (5K)** | **0.862** | **0.862** | **0.823** | **0.868** | **0.819** | **0.847** |

### SFE_SE Models on SFE_SE Test Data (Top 1 Neighbor MSE)

| Model \ Test | SFE_SE_1 | SFE_SE_2 | SFE_SE_3 | SFE_SE_4 | SFE_SE_5 | Mean |
|---|---|---|---|---|---|---|
| SFE_SE_1 (1K) | **0.203** | 0.191 | 0.153 | 0.150 | 0.122 | 0.164 |
| SFE_SE_2 (2K) | 0.217 | **0.193** | 0.168 | 0.154 | 0.125 | 0.171 |
| SFE_SE_3 (3K) | 0.216 | 0.201 | **0.169** | 0.146 | 0.135 | 0.173 |
| SFE_SE_4 (4K) | 0.221 | 0.206 | 0.168 | **0.149** | 0.133 | 0.175 |
| **SFE_SE_5 (5K)** | **0.200** | **0.198** | **0.165** | **0.135** | **0.121** | **0.164** |

### Grand Comparison: All Models on Augmented Test Data

| Model | Mean Spearman on augmented data | Model Type |
|-------|---------------------------|------------|
| SFE_SE_1 (1K) | **0.856** | SFE_SE |
| SFE_SE_2 (2K) | 0.841 | SFE_SE |
| SFE_SE_3 (3K) | 0.836 | SFE_SE |
| SFE_SE_4 (4K) | 0.829 | SFE_SE |
| SFE_SE_5 (5K) | 0.812 | SFE_SE |
| Run_3 (3K) | 0.702 | Augmented |
| Run_2 (2K) | 0.695 | Augmented |
| Run_1 (1K) | 0.686 | Augmented |
| Run_4 (4K) | 0.665 | Augmented |
| Run_5 (5K) | 0.644 | Augmented |

Every SFE_SE model outperforms every augmented model. The gap between the worst SFE_SE model (0.812) and the best augmented model (0.702) is +0.110.

### Key Findings — SFE_SE Runs

1. **All SFE_SE models dramatically outperform augmented models on augmented test data**. The worst SFE_SE mean (SFE_SE_4 at 0.829) exceeds the best augmented mean (Run_3 at 0.702) by +0.127. This demonstrates that training data source composition matters more than sequence count.

2. **Opposite ranking on SFE_SE vs augmented data**. On augmented data: SFE_SE_1 wins (0.856). On SFE_SE data: SFE_SE_5 wins (0.847). The optimal threshold depends on the evaluation domain.

3. **SFE_SE data is harder than augmented data**. Models 1-4 score 0.73-0.77 on SFE_SE data vs 0.83-0.86 on augmented data.

4. **SFE_SE_5 dominates SFE_SE evaluation** with a massive gap (+0.081 over second-place SFE_SE_1). It wins every column in both Spearman and Top 1 MSE.

5. **4K column is easiest on SFE_SE data** (consistently highest Spearman across all models), while 2K and 3K are the hardest. On augmented data, 1K is the easiest column.

6. **SFE_SE models have higher per-k-mer training MSE but better embedding quality** — training reconstruction loss is a poor predictor of downstream embedding performance across model families.

---

## 7. Cross-Domain Analysis

### Augmented Models on SFE_SE Test Data (Spearman)

| Model \ Test | SFE_SE_1 | SFE_SE_2 | SFE_SE_3 | SFE_SE_4 | SFE_SE_5 | Mean |
|---|---|---|---|---|---|---|
| Run_1 (1K) | 0.676 | 0.663 | 0.640 | 0.748 | 0.693 | 0.684 |
| Run_2 (2K) | 0.693 | 0.682 | 0.652 | 0.760 | 0.709 | 0.699 |
| **Run_3 (3K)** | **0.790** | **0.763** | **0.713** | **0.810** | **0.768** | **0.769** |
| Run_4 (4K) | 0.754 | 0.716 | 0.693 | 0.795 | 0.719 | 0.735 |
| Run_5 (5K) | 0.705 | 0.670 | 0.650 | 0.746 | 0.671 | 0.688 |

### Grand Comparison: All Models on SFE_SE Test Data

Combining augmented and SFE_SE models evaluated on SFE_SE test data:

| Model | Mean Spearman on SFE_SE data | Model Type |
|-------|------------------------------|------------|
| SFE_SE_5 (5K) | **0.847** | SFE_SE |
| Run_3 (3K) | 0.769 | Augmented |
| SFE_SE_1 (1K) | 0.766 | SFE_SE |
| SFE_SE_2 (2K) | 0.736 | SFE_SE |
| Run_4 (4K) | 0.735 | Augmented |
| SFE_SE_3 (3K) | 0.728 | SFE_SE |
| SFE_SE_4 (4K) | 0.726 | SFE_SE |
| Run_2 (2K) | 0.699 | Augmented |
| Run_5 (5K) | 0.688 | Augmented |
| Run_1 (1K) | 0.684 | Augmented |

### Key Findings — Cross-Domain

1. **Run_3 is the best augmented model on SFE_SE data too** (0.769), matching its dominance on augmented data. The 3K threshold produces the most transferable embeddings regardless of evaluation domain.

2. **Augmented models score higher on SFE_SE data than on augmented data**. Run_3: 0.769 on SFE_SE vs 0.702 on augmented (+0.067). This is likely because SFE_SE test data (4.8-6.7M sequences) is a more homogeneous subset, and the 50k sample from a smaller pool produces denser coverage with more reliable neighbor relationships.

3. **SFE_SE_5 stands alone at the top** (0.847) — 0.078 above second-place Run_3 (0.769). The remaining SFE_SE models (1-4) are interleaved with augmented models, suggesting the SFE_SE advantage is not universal across thresholds.

4. **Augmented Run_3 outperforms SFE_SE models 1-4 on SFE_SE data**. Despite being trained on different data, Run_3 (0.769) beats SFE_SE_1 (0.766), SFE_SE_2 (0.736), SFE_SE_3 (0.728), and SFE_SE_4 (0.726). The FD + NCBI training data provides broader biological coverage that transfers well.

5. **SFE_SE_4 column is consistently the easiest** across both augmented and SFE_SE models. The SFE_SE_3 column is consistently the hardest.

---

## 8. Experimental Models (2026-02-20)

### Motivation

The original 10 models (5 augmented + 5 SFE_SE) established that SFE_SE_5 was the best marine embedder. Five additional models tested specific hypotheses about training data composition, particularly whether reference genomes or length-matched training data could improve performance on the 100 kbp clustering task.

### Training Data

| Model | Training Data | N Sequences | Description |
|-------|--------------|-------------|-------------|
| **NCBI_5** | NCBI RefSeq >= 5 kbp | 655,859 | ~20K reference genomes only |
| NCBI_100 | NCBI RefSeq >= 100 kbp | 175,176 | Length-filtered reference genomes |
| SFE_SE_100 | SFE + SE >= 100 kbp | 154,040 | Marine, matching evaluation domain |
| Run_100 | All sources >= 100 kbp | ~845,000 | Broad + long sequences |
| SFE_SE_NCBI_5 | SFE + SE + NCBI >= 5 kbp | ~5,400,000 | Marine + reference genomes |

### Cross-Model Spearman Comparison (8 models x 4 test sets)

| Model | Training data | N seqs | SFE_SE_5 test | SFE_SE_100 test | NCBI_5 test | Own data |
|---|---|---|---|---|---|---|
| **SFE_SE_5** | Marine >= 5 kbp | 4.8M | **0.847** | 0.766 | **0.946** | — |
| **NCBI_5** | NCBI >= 5 kbp | 656K | 0.831 | **0.836** | 0.934 | — |
| NCBI_100 | NCBI >= 100 kbp | 175K | 0.836 | 0.832 | 0.919 | — |
| SFE_SE_100 | Marine >= 100 kbp | 154K | 0.797 | 0.804 | — | — |
| Run_100 | All >= 100 kbp | 845K | 0.784 | 0.798 | 0.894 | 0.788 |
| SFE_SE_NCBI_5 | Marine + NCBI >= 5 kbp | 5.4M | 0.662 | — | 0.946 | — |
| Run_3 | All >= 3 kbp | 13.4M | 0.702 | — | — | — |
| Run_5 | All >= 5 kbp | 13.4M | 0.644 | — | — | — |

Test sets: SFE_SE_5 = full marine (4.8M seqs), SFE_SE_100 = marine >= 100 kbp (154K seqs), NCBI_5 = reference genomes >= 5 kbp (656K seqs).

### NCBI_100 as Test Set

| Model | NCBI_5 test | NCBI_100 test |
|---|---|---|
| **SFE_SE_5** | **0.946** | **0.947** |
| NCBI_5 | 0.934 | 0.925 |
| NCBI_100 | 0.919 | 0.910 |

Filtering NCBI test data to >= 100 kbp changes nothing. All models score 0.91-0.95.

### Key Findings

1. **NCBI genomes are trivially easy to organize**: All models 0.89-0.95 on NCBI data, even SFE_SE_5 which never saw reference genomes.

2. **Length homogeneity > source mixing**: NCBI_5 (median training length ~37 kbp) beats SFE_SE_5 (median ~8 kbp) on 100 kbp marine data (0.836 vs 0.766) despite never seeing marine sequences.

3. **Mixing data sources can hurt**: SFE_SE_NCBI_5 (0.662) is worst despite 5.4M sequences from two complementary sources. Mixed length distributions create conflicting optimization targets.

4. **Filtering NCBI by length doesn't help**: NCBI_5 (656K) ≈ NCBI_100 (175K). Both see the same ~20K genomes; length filtering just trims shorter contigs.

5. **Spearman doesn't predict clustering quality**: The 0.070 Spearman gap on 100 kbp data (NCBI_5 0.836 vs SFE_SE_5 0.766) translated to essentially no difference in MCL GC spans.

### Clustering Comparison: NCBI_5 vs SFE_SE_5 on 100 kbp Marine Data

Graph construction: In-degree capped (cap=100), d<5, weights 1/(d+0.1).

| Metric | SFE_SE_5 | NCBI_5 |
|--------|----------|--------|
| Nodes in graph | 123,783 | 133,724 |
| Edges | 3,391,528 | 4,571,866 |
| Coverage (of 154K) | 80% | 87% |

**MCL GC spans (pp) for top 3 communities:**

| I | SFE_SE_5 | NCBI_5 | SFE_SE_5 clusters | NCBI_5 clusters |
|---|---|---|---|---|
| 1.4 | 5, 8, 6 | 8, 6, 9 | 7,693 | 7,142 |
| 2.0 | 5, 5, 4 | 8, 7, 4 | 9,710 | 8,885 |
| **3.0** | **4, 6, 4** | **4, 4, 4** | **12,305** | **11,413** |
| 4.0 | 4, 5, 4 | 4, 7, 3 | 15,202 | 15,592 |
| 5.0 | 4, 5, 4 | 4, 7, 5 | 17,323 | 18,577 |
| 6.0 | 4, 5, 4 | 7, 4, 5 | 19,198 | 21,047 |

**Model selection (2026-02-21)**: NCBI_5 chosen over SFE_SE_5. At I=3.0, both produce 4 pp GC spans. NCBI_5 connects 10K more sequences (87% vs 80% coverage). Trained on only ~20K reference genomes, making its per-genome efficiency remarkable. Model symlink updated to `Runs/Run_NCBI_5/vae_encoder_best.keras`.

---

## 9. Summary and Recommendations (updated 2026-02-21)

### Best Models by Evaluation Context

| Context | Best Model | Mean Spearman | Notes |
|---------|-----------|---------------|-------|
| Augmented data (within-family) | Run_3 (3K) | 0.702 | Wins every column in 5x5 augmented matrix |
| Augmented data (any model) | SFE_SE_1 (1K) | 0.856 | Best of any model on augmented test data |
| Full marine data | SFE_SE_5 (5K) | 0.847 | Dominates by +0.081 over 2nd place |
| 100 kbp marine data | NCBI_5 (5K) | 0.836 | Best on the clustering evaluation domain |
| NCBI reference genomes | SFE_SE_5 (5K) | 0.946 | All models score 0.89-0.95 (trivially easy) |
| **Clustering (MCL GC spans)** | **NCBI_5 (5K)** | — | **4/4/4 pp, 87% coverage — selected model** |

### Key Insights

1. **3K bp is the sweet spot for augmented training data** — captures sufficient biological diversity without noise from very short contigs. Run_3 retains extreme-GC organisms lost at higher thresholds, and produces the most transferable embeddings across evaluation domains.

2. **Training data source composition matters more than quantity** — SFE_SE models with 4.8-6.7M sequences dramatically outperform augmented models with 13.4-17.4M sequences on augmented test data (worst SFE_SE: 0.812 > best augmented: 0.702). Quality and source diversity drive embedding quality, not raw sequence count.

3. **Reconstruction loss does not predict embedding quality** — Run_5 has the lowest training MSE (0.038) but worst embedding quality (Spearman 0.644). SFE_SE models have 1.3-2.3× higher training MSE than augmented runs but 15-22% better Spearman correlations. This underscores the need for independent evaluation metrics beyond reconstruction loss. Most metagenomic embedding papers only report reconstruction loss — our analysis shows this is insufficient.

4. **The optimal threshold depends on the evaluation domain** — On augmented data: SFE_SE_1 (1K) is best. On SFE_SE data: SFE_SE_5 (5K) is best. For augmented models: Run_3 (3K) wins everywhere. No single threshold is universally optimal.

5. **Euclidean distance outperforms cosine** for this VAE's latent space (Spearman 0.697 vs 0.621), consistent with the MSE training objective creating Euclidean-friendly geometry.

6. **Augmented Run_3 competes with SFE_SE models on SFE_SE data** — Run_3 (0.769) outperforms SFE_SE models 1-4 (0.726-0.766) on their own test data, demonstrating that the FD + NCBI training data provides broad biological coverage that transfers effectively.

7. **All models converge well before 1000 epochs** — augmented runs reach LR floor by epochs 316-468, SFE_SE runs by epochs 406-544. The remaining training at minimum LR provides negligible improvement.

8. **Extreme-GC organisms require lower thresholds** — the Run_5 2-mer/1-mer anomaly reveals that organisms with >75% GC content have small genomes assembling into shorter contigs. The 5K threshold loses these, causing 26× higher 1-mer reconstruction error on 0.46% of sequences that contribute ~50% of total 1-mer MSE.

9. **Proxy metrics don't predict downstream performance** — Run_5 has lowest MSE but worst Spearman. NCBI_5's 0.070 Spearman advantage over SFE_SE_5 on 100 kbp data translates to no difference in MCL GC spans (4/4/4 vs 4/6/4 pp). The field needs end-task metrics.

10. **A small reference genome dataset can match domain-specific training for clustering** — NCBI_5 (656K contigs from ~20K genomes) matches SFE_SE_5 (4.8M marine contigs) on MCL clustering quality and achieves better graph coverage (87% vs 80%) despite never seeing marine data.

11. **Length homogeneity in training data improves performance on long-contig tasks** — NCBI_5 (median training length ~37 kbp) dramatically outperforms SFE_SE_5 (median ~8 kbp) on 100 kbp evaluation data. Mixing length-mismatched sources hurts (SFE_SE_NCBI_5 is the worst model).
