# Claude Session Notes

## About This File

Torben collaborates with multiple Claude instances:
- **This host (Threadripper)**: Heavy-duty computation - VAE training, large dataset processing, clustering
- **Desktop**: Writing and lighter tasks

These notes are shared across both so each instance can understand context from the other's sessions. Keep notes clear and comprehensive.

## Related Repositories
- **Embedding** (`tnn111/Embedding`, public): This repo - VAE training, inference, k-mer calculation
- **ClusteringPaper** (`tnn111/ClusteringPaper`, private): Nature Methods paper being written by sibling Claude instance on desktop
  - Cloned locally to `/home/torben/ClusteringPaper/`
  - Contains: Introduction.md, Methods_VAE.md, Results_VAE.md, Draft.md, Data.md, References.md, Paper.md
  - Latest commit: `ed11249` - Retarget paper to Nature Methods

---

# Consolidated Project Review (2026-02-19)

This section is the definitive reference for the sibling Claude instance writing the paper. It supersedes the chronological session notes archived below.

## 1. Architecture and Training

### VAE Architecture
```
Encoder: 2772 -> Dense(1024) -> BN -> LeakyReLU(0.2)
         -> Dense(512) -> BN -> LeakyReLU(0.2)
         -> z_mean(384) / z_log_var(384)

Decoder: 384 -> Dense(512) -> BN -> LeakyReLU(0.2)
         -> Dense(1024) -> BN -> LeakyReLU(0.2)
         -> Dense(2772, sigmoid)

Total parameters: ~7.1M (Encoder: 3.6M, Decoder: 3.5M)
```

### Input Features (2,772 dimensions)
All canonical k-mer frequencies from 6-mer through 1-mer. Each k-mer size group is independently normalized to sum to 1.0 by `calculate_kmer_frequencies`.

| Group | Features | Canonical k-mers |
|-------|----------|-----------------|
| 6-mer | 2,080 | of 4,096 total |
| 5-mer | 512 | of 1,024 total |
| 4-mer | 136 | of 256 total |
| 3-mer | 32 | of 64 total |
| 2-mer | 10 | of 16 total |
| 1-mer | 2 | of 4 total |

### Loss Function
- **Transform**: Per-group CLR (Centered Log-Ratio) applied independently to each k-mer size group, respecting compositional independence
- **Pseudocount**: Jeffreys prior (0.5 / n_features per group) - mathematically principled uninformative prior for multinomial data
- **Loss**: MSE on CLR-transformed features + beta * KL divergence
- **Beta**: 0.05 (beta-VAE for clustering-friendly latent space)
- **KL warmup**: 5 epochs (skipped on resume)

### Key Design Decisions
- **384-dim latent space**: Increased from 256 after finding 6-mers dominated 95% of error (13% 6-mer MSE improvement: 1.56 -> 1.35)
- **Beta = 0.05**: Systematic sweep showed best balance of reconstruction + regularization + clustering quality
- **Per-group CLR**: Each k-mer size is a separate composition; joint CLR mixed them incorrectly (geometric mean dominated by 2,080 six-mers)
- **Jeffreys prior**: Replaced arbitrary 1e-6 pseudocount; reduced zero-value gap from ~6 to ~0.7 log units
- **No dropout**: Healthy data-to-param ratio (13.4M/7M), KL regularization sufficient, train/val gap < 1 pt

### Training Procedure
- Optimizer: Adam, initial LR 1e-4
- LR schedule: ReduceLROnPlateau (floor 1e-6, patience 10, factor 0.5)
- Batch size: 1024
- Epochs: 1000 (converges by ~500; LR hits floor by epoch 316-468)
- 90/10 train/val split on pre-shuffled data
- JAX backend for Keras

## 2. Data Sources

| Source | Description | Sequences (1K bp) |
|--------|-------------|-------------------|
| **SFE** | San Francisco Estuary metagenomes | ~3.4M |
| **SE** | Baltic Sea metagenomes | ~3.3M |
| **FD** | Microflora Danica (aquatic + soil) | ~8.0M |
| **NCBI** | RefSeq representative genomes | ~0.7M |

Two model families trained:
- **Augmented** (FD + NCBI + SFE + SE): 13.4-17.4M sequences, 4 sources
- **SFE_SE** (SFE + SE only): 4.8-6.7M sequences, 2 marine sources

Five length thresholds per family: 1K, 2K, 3K, 4K, 5K bp minimum.

## 3. Model Selection

### Evaluation Metric
**Spearman correlation** between pairwise latent-space Euclidean distance and pairwise input-space k-mer MSE, measured on 50K validation samples (100 queries x 50 neighbors + random baseline). This validates that latent distance preserves biological similarity ordering. Reconstruction loss alone is insufficient - Run_4 had lowest MSE but worst Spearman.

### Full Cross-Comparison Results

**Augmented 5x5 (model x test data), Spearman correlation:**

| Model | 1K | 2K | 3K | 4K | 5K | Mean |
|-------|------|------|------|------|------|------|
| Run_1 (1K) | 0.751 | 0.616 | 0.723 | 0.703 | 0.635 | 0.686 |
| Run_2 (2K) | 0.764 | 0.627 | 0.729 | 0.711 | 0.643 | 0.695 |
| **Run_3 (3K)** | **0.769** | **0.639** | **0.721** | **0.722** | **0.660** | **0.702** |
| Run_4 (4K) | 0.738 | 0.598 | 0.692 | 0.674 | 0.625 | 0.665 |
| Run_5 (5K) | 0.726 | 0.584 | 0.655 | 0.640 | 0.616 | 0.644 |

**SFE_SE models on augmented test data:**

| Model | Mean Spearman |
|-------|--------------|
| SFE_SE_1 (1K) | **0.856** |
| SFE_SE_2 (2K) | 0.841 |
| SFE_SE_3 (3K) | 0.836 |
| SFE_SE_4 (4K) | 0.829 |
| SFE_SE_5 (5K) | 0.812 |

**SFE_SE models on SFE_SE test data:**

| Model | Mean Spearman |
|-------|--------------|
| **SFE_SE_5 (5K)** | **0.847** |
| SFE_SE_1 (1K) | 0.766 |
| SFE_SE_2 (2K) | 0.736 |
| SFE_SE_3 (3K) | 0.728 |
| SFE_SE_4 (4K) | 0.726 |

### Model Rankings by Use Case

| Context | Best Model | Spearman |
|---------|-----------|----------|
| Augmented data (within-domain) | Run_3 (3K) | 0.702 |
| Augmented data (any model) | SFE_SE_1 (1K) | 0.856 |
| SFE_SE data (within-domain) | SFE_SE_5 (5K) | 0.847 |
| Cross-domain generalist | Run_3 (3K) | 0.702 / 0.769 |

### Selected Model for Clustering: SFE_SE_5
- Dominates on SFE_SE test data (+0.081 over second place)
- Specializes to marine sequences (the target domain)
- 5K bp threshold captures longer contigs with cleaner k-mer profiles

## 4. Key Scientific Findings

### 4.1 Training Data Composition > Quantity
**The most striking finding.** SFE_SE models (4.8-6.7M sequences, 2 sources) dramatically outperform augmented models (13.4-17.4M sequences, 4 sources) - even on augmented test data the SFE_SE models never saw.

- Worst SFE_SE (0.812) beats best augmented (0.702): gap of +0.110
- Interpretation: spreading 384-dim latent space across wider biology (FD + NCBI + marine) dilutes local distance structure
- **Focused, high-quality data > broad, noisy data** for embedding quality

### 4.2 Reconstruction Loss is Insufficient for Evaluation

| Model | Training MSE | Embedding Spearman |
|-------|-------------|-------------------|
| Run_5 (5K) | **0.038** (lowest) | 0.644 (worst) |
| Run_3 (3K) | 0.052 | 0.702 (best augmented) |
| SFE_SE_5 (5K) | 0.049 | 0.847 (best overall) |

SFE_SE models have 1.3-2.3x higher training MSE but 15-22% better Spearman. Most metagenomic embedding papers only report reconstruction loss.

### 4.3 Euclidean > Cosine for Latent Distance
- Euclidean Spearman: **0.697**
- Cosine Spearman: 0.621
- Gap: +0.076
- Expected because MSE loss creates Euclidean-friendly geometry
- Cosine collapses in 384-dim (std 0.061, no discriminative range)

### 4.4 Train/Val Gap was Data Shuffling, NOT BatchNorm
Runs 4-5 initially showed ~34-48 pt train/val gaps. Root cause: `concatenate_matrices` stacked data without shuffling, making validation unrepresentative. With proper shuffling, gaps dropped to < 1 pt. The BatchNorm hypothesis was wrong - gap existed from epoch 1 (before memorization possible).

### 4.5 GC Content Dominates Latent Space
t-SNE of 4.8M embeddings shows two major lobes: low-GC (20-40%) and high-GC (50-70%). GC is the primary axis of variation because it's the strongest compositional signal in k-mer frequencies.

### 4.6 Extreme-GC Anomaly at High Length Thresholds
Run_5 (5K bp) shows 26x higher 1-mer error on sequences >75% GC (0.46% of data contributing ~50% of 1-mer MSE). These organisms have small genomes assembling into shorter contigs, making them underrepresented at high thresholds. Run_3's 3K threshold retains more examples.

### 4.7 Low-Dimensional Manifold Structure
Count-vs-distance analysis shows linear growth of neighbors with radius (R^2 = 0.954) for r < 0.5 in latent space. In 384-dim, volume grows as r^383. Linear growth confirms data lies on a low-dimensional manifold. TWO-NN estimates:
- Full dataset: d-hat ~ 9
- 100 kbp filtered: d-hat = 3.74
- 50 kbp filtered: d-hat = 4.42

## 5. Clustering Analysis

### 5.1 The Archipelago Model
The latent space is **discrete, not continuous**. Each species/lineage occupies a tiny dense island surrounded by empty space:
- At 10 kbp: 74.8% of sequences have NO neighbor within Euclidean d=10
- At 100 kbp: 82% have a neighbor within d=5
- The "rare biosphere" (low-abundance, uncultured lineages) explains the singletons

### 5.2 The Hub Problem
In-degree analysis of kNN graphs reveals massive hubs:
- Top hub: in-degree 58,377 (2% of dataset pointing to one sequence)
- Top 20 hubs: all ~28-29% GC, 400 kbp - 1.2 Mbp (large marine genomes)
- Median in-degree: 0
- Hubs act as transitivity chain anchors: A -> hub -> B merges unrelated sequences

### 5.3 Graph Construction Methods

Three approaches tested at each distance threshold:

| Method | Mechanism | Pros | Cons |
|--------|-----------|------|------|
| **Symmetric kNN** | Union of both directions | Maximum coverage | Hubs create giant components |
| **Mutual kNN** | Intersection (both must agree) | Eliminates hubs completely | 85% isolated, crippling coverage |
| **In-degree capped (cap=100)** | Limit max in-degree | Reduces hubs while preserving coverage | Middle ground |

In-degree capping is the best compromise: reduces largest community from 119K to 56K while maintaining ~47% coverage (at 10 kbp d=10). The cap=100 parameter was chosen as well above the P99 of in-degree in dense regions.

### 5.4 MCL vs Leiden

MCL dramatically outperforms Leiden on dense graphs by breaking transitivity chains through flow-based clustering.

**GC spans (pp) for top 3 communities at 10 kbp, d=10 capped graph:**

| Method | C1 | C2 | C3 |
|--------|----|----|-----|
| Leiden (capped) | 42-49 | 42-49 | 42-49 |
| MCL I=3.0 | 9 | 7 | 7 |
| MCL I=5.0 | 9 | 7 | 6 |

**Why MCL wins**: Leiden merges anything reachable through any path (modularity optimization). MCL's iterative expansion + inflation naturally attenuates weak connections through flow simulation.

### 5.5 Weight Functions for MCL

Tested on 100 kbp d=7 and d=10 graphs:
- **1/(d+0.1)**: Works well - gentle similarity, lets inflation do the work
- **exp(-d)**: Much worse - over-suppresses medium-range edges (d=3-5) that MCL needs for cluster boundary definition

The counterintuitive result: gentler similarity functions outperform exponential because MCL needs flow to traverse the local neighborhood before inflation cuts weak connections.

### 5.6 Length Threshold Analysis

#### nn1 Sweep (10K queries per threshold, against ALL sequences)

| Threshold | Sequences | Mean nn1 | Median nn1 | nn1 < 5 |
|-----------|-----------|----------|------------|---------|
| 3 kbp | 5,951,176 | 12.76 | 12.28 | 9.5% |
| 5 kbp | 4,776,770 | 11.43 | 11.17 | 10.0% |
| 10 kbp | 3,039,927 | 9.27 | 9.20 | 13.8% |
| 20 kbp | 1,556,556 | 7.11 | 7.02 | 24.8% |
| **50 kbp** | **461,674** | **4.99** | **4.82** | **53.4%** |
| **100 kbp** | **154,040** | **3.58** | **3.38** | **81.6%** |
| 200 kbp | 53,138 | 2.64 | 2.36 | 92.8% |
| 500 kbp | 11,006 | 2.08 | 1.70 | 93.4% |

**The knee is at 50-100 kbp.** Below 50 kbp: noise-dominated. Above 200 kbp: diminishing returns with shrinking dataset.

#### Clustering Quality by Length Threshold

**MCL GC spans (pp) for top 3 communities at best tested configuration:**

| Threshold | Config | C1 | C2 | C3 | Verdict |
|-----------|--------|----|----|-----|---------|
| 10 kbp | d=5, I=2.0 | 5 | 5 | 5 | Good quality, 12% coverage |
| 10 kbp | d=7, I=3.0 | 6 | 4 | 7 | Good quality, 24% coverage |
| 10 kbp | d=10, I=3.0 | 9 | 7 | 7 | Acceptable, 47% coverage |
| 50 kbp | d=7, I=3.0 | 6 | 8 | 5 | Inconsistent (some 9-10 pp) |
| **100 kbp** | **d=5, I=3.0** | **4** | **6** | **5** | **Consistently tight** |
| 100 kbp | d=5, I=5.0 | 6 | 6 | 4 | Tight, smaller clusters |

### 5.7 100 kbp: The Gold Standard

At 100 kbp with d=5, MCL produces:
- **154,040 sequences**, 80.7% clustered (symmetric Leiden), ~80% connected at d=5
- **Intrinsic dimensionality**: d-hat = 3.74 (clean, low-dimensional manifold)
- **MCL I=3.0**: 12,305 clusters, largest 182, GC spans 4-6 pp
- **MCL I=1.4**: 7,693 clusters, largest 1,056, GC spans 7-9 pp
- All three Leiden variants agree broadly (symmetric/mutual/capped produce similar structure)
- Even Leiden achieves reasonable GC spans (7-15 pp) - transitivity chains are minimal at this threshold

### 5.8 50 kbp: Marginal

At 50 kbp with d=7:
- **461,674 sequences** (3x more), but nn1 median = 5.01 (right at the threshold)
- Leiden fails badly: GC spans 19-46 pp (transitivity chains dominate)
- MCL partially rescues: 4-10 pp, but inconsistent across communities
- **The relative threshold argument (d/median_nn1) did NOT hold** - absolute edge quality matters. Edges at d=6-7 carry less biological signal than edges at d=3-5, regardless of local context.
- **Conclusion**: Not worth the quality tradeoff. 154K sequences at 100 kbp is sufficient.

### 5.9 10 kbp: Noise Floor

At 10 kbp with d=10:
- **3,039,927 sequences**, but only 49.2% clustered, d-hat = 8.99
- MCL at d=5: only 12% coverage but excellent quality (5 pp GC)
- MCL at d=10: 47% coverage, 7-9 pp GC (acceptable but noisier)
- Length filtering removes singletons (50.8% -> 19.3% at 100 kbp) while barely affecting clustered count

### 5.10 Distance Threshold Experiments at 100 kbp

**d=5 vs d=7 (MCL, 1/(d+0.1) weights, capped graph):**

| d | I=1.4 GC spans | I=3.0 GC spans | Verdict |
|---|----------------|----------------|---------|
| d=5 | 7, 9, 9 | 4, 6, 5 | **Best tested so far** |
| d=7 | 13, 12, 18 | 4, 5, 5 | Comparable at I=3.0 but worse at I=1.4 |
| d=7 exp(-d) | 21, 18, 14 | 8, 7, 6 | Worse (exponential over-suppresses medium edges) |

**d=5 is the best tested distance threshold** for 100 kbp data. Matches the natural neighborhood structure (mean nn1 = 3.69, 78% connected at d=5).

## 6. Codebase Status

### Scripts

| File | Purpose | Status |
|------|---------|--------|
| `VAE.py` | Training script | Production-ready |
| `embedding` | Inference (PEP 723 standalone) | Fixed: uses z_mean |
| `create_and_load_db` | ChromaDB loader (PEP 723) | Fixed: uses L2 distance, z_mean |
| `calculate_kmer_frequencies` | K-mer calculator from FASTA | Production-ready |
| `concatenate_matrices` | Memory-efficient matrix merge | Fixed: shuffle support |
| `verify_local_distances.py` | Latent space quality validation | Production-ready |
| `verify_knn_quality` | Alternate validation | Functional |
| `VAE_noGC.py` | Ablation variant (no 1-mers) | Experimental |

### Notebooks

| Notebook | Purpose | Status |
|----------|---------|--------|
| `clustering_100.ipynb` | 100 kbp analysis (gold standard) | Complete |
| `clustering_050.ipynb` | 50 kbp analysis (marginal) | Complete |
| `clustering_010.ipynb` | 10 kbp analysis (noise floor) | Complete |
| `clustering.ipynb` | Original exploration (full dataset) | Superseded by above |
| `shrub_of_life.ipynb` | ChromaDB query exploration | Historical |

### Model Artifacts

| Location | Model | Purpose |
|----------|-------|---------|
| `Runs/Run_3/` | Best augmented (3K bp) | General-purpose embedding |
| `Runs/Run_SFE_SE_5/` | Best marine (5K bp) | Clustering target |
| `vae_encoder_best.keras` | Symlink -> Run_SFE_SE_5 | Current default |

### Known Remaining Issues

1. **Custom layers duplicated across 6 files** - `Sampling`, `ClipLayer`, `clr_transform` copy-pasted into VAE.py, VAE_noGC.py, embedding, create_and_load_db, verify_local_distances.py, verify_knn_quality. Should extract to shared module.
2. **Training history overwritten on resume** - `vae_history.pkl` loses full curve.
3. **pyproject.toml name is "clustering"** - should be "embedding" or "vae".
4. **README.md is empty**.

### Previously Fixed Issues (verified 2026-02-19)
- Inference now uses z_mean (both `embedding` and `create_and_load_db`)
- ChromaDB now uses L2 (Euclidean) distance
- Model symlink now points to Run_SFE_SE_5
- `main.py` and `convert_txt_to_npy` removed
- `verify_local_distances.py` column indices updated
- Per-group CLR with Jeffreys prior implemented
- Data shuffling fixed in `concatenate_matrices`

## 7. Data Inventory (Runs/)

### Training Data
- `kmers_1.npy` through `kmers_5.npy`: Augmented data (139-180 GB each)
- `kmers_SFE_SE_1.npy` through `kmers_SFE_SE_5.npy`: Marine data (50-70 GB each)
- Corresponding `ids_*.txt` files

### Analysis Artifacts (100 kbp pipeline)
- `embeddings_SFE_SE_100.npy`: 154K x 385 (227 MB)
- `ids_SFE_SE_100.txt`: 154K IDs (2.3 MB)
- `neighbors_SFE_SE_100.tsv`: k=50 neighbors (172 MB)
- `graph_capped100_d5_100kbp.tsv`: MCL input graph (100 MB)

### Analysis Artifacts (50 kbp pipeline)
- `embeddings_SFE_SE_050.npy`: 462K x 385 (679 MB)
- `ids_SFE_SE_050.txt`: 462K IDs (6.7 MB)
- `neighbors_SFE_SE_050.tsv`: k=50 neighbors (514 MB)
- `graph_capped100_d7_50kbp.tsv`: MCL input graph (298 MB)

### MCL Results
- `MCL_100kbp_d5/`: Best configuration (6 inflation values)
- `MCL_100kbp_d7/`: d=7 experiment (worse than d=5)
- `MCL_100kbp_d7_exp/`: d=7 with exp(-d) weights (worst)
- `MCL_50kbp_d7/`: 50 kbp experiment (marginal)
- `Archive/`: Older MCL runs on 10 kbp data (MCL_d5, MCL_d7, MCL_d10, etc.)

## 8. References

49 references organized by 35 claims in `Claude_References.md`. Key categories:
- **Foundational**: VAE (Kingma & Welling 2014), beta-VAE (Higgins et al. 2017)
- **Compositional data**: CLR (Aitchison 1982, 1986), Jeffreys prior (Jeffreys 1946), microbiome (Gloor et al. 2017)
- **K-mer signatures**: Karlin & Burge 1995, Pride et al. 2003
- **Related methods**: VAMB (Nissen et al. 2021), MetaBAT (Kang et al. 2015, 2019)
- **Graph clustering**: MCL (van Dongen 2000), Leiden (Traag et al. 2019)
- **Hubness**: Radovanovic et al. 2010, Tomasev et al. 2014
- **Rare biosphere**: Sogin et al. 2006, Lynch & Neufeld 2015
- **Dimensionality**: TWO-NN (Facco et al. 2017), distance concentration (Beyer et al. 1999)
- **Long-read metagenomics**: metaMDBG (Benoit et al. 2024), HiFi (Kim et al. 2022)

## 9. Conclusions for the Paper

### Main Claims (with supporting evidence)

1. **Multi-scale k-mer VAE produces high-quality embeddings for metagenomics.** 384-dim latent space preserves biological similarity ordering (Spearman 0.70-0.85 between latent distance and k-mer MSE). Local metric structure is geometrically sound (linear count-vs-distance growth, R^2 = 0.954).

2. **Training data composition matters more than quantity.** SFE_SE models (4.8-6.7M seqs) beat augmented models (13.4-17.4M seqs) by +0.110 Spearman even on augmented test data. Focused domain-specific training > broad generalist training.

3. **Reconstruction loss is insufficient for evaluating embedding quality.** Run_5 has lowest MSE but worst Spearman. SFE_SE models have 1.3-2.3x higher MSE but 15-22% better embedding quality. The field needs latent space metrics.

4. **Euclidean distance outperforms cosine in the VAE latent space** (+0.076 Spearman). Expected from MSE loss geometry, but contradicts common practice (most vector DBs default to cosine).

5. **Sequence length is the primary driver of clustering quality.** At >= 100 kbp, even simple methods work (Leiden achieves 7-15 pp GC spans). Below 10 kbp, even MCL struggles. The nn1 sweep shows a clear knee at 50-100 kbp.

6. **MCL dramatically outperforms Leiden on dense metagenome graphs.** GC spans: MCL 4-9 pp vs Leiden 19-49 pp at same threshold. Flow-based clustering breaks transitivity chains that modularity optimization cannot.

7. **Hub nodes cause giant component formation.** Top hub has in-degree 58,377 (2% of dataset). In-degree capping (cap=100) is an effective compromise between symmetric (hub-dominated) and mutual (coverage-crippling) graph construction.

8. **The latent space has archipelago structure** reflecting real biological discreteness. Most sequences are isolated; clusterable sequences form distinct islands. This is not an embedding failure - it reflects the rare biosphere and k-mer profile noise on short contigs.

### Best Configuration
- **Model**: SFE_SE_5 (5K bp threshold, marine-focused)
- **Distance metric**: Euclidean (L2)
- **Length threshold**: >= 100 kbp
- **Graph construction**: In-degree capped (cap=100), distance threshold d=5, weights 1/(d+0.1)
- **Clustering**: MCL with inflation I=3.0 (12,305 clusters, largest 182, GC spans 4-6 pp)

### What We Have NOT Established
- Whether 100 kbp / d=5 is truly optimal (it's the best tested so far; other combinations in the 50-200 kbp range remain unexplored)
- Whether these findings generalize to non-marine metagenomes
- Whether the specific GC span numbers hold for different assemblers or sequencing platforms
- Biological validation beyond GC content (taxonomy, functional annotation)

### Remaining Work
- Cross-method validation (clusters stable across both Leiden and MCL = high confidence)
- Biological characterization of clusters (taxonomy, function, source composition)
- Update sweep plot in clustering_010.ipynb Cell 23

---

# Archived Session Notes (Chronological)

The following are historical session notes preserved for reference. The consolidated review above supersedes these.

## 2026-02-02: Clustering analysis and codebase cleanup

### HDBSCAN Clustering on t-SNE
- Ran HDBSCAN clustering on full 4.8M t-SNE coordinates
- Parameters: `min_cluster_size=1000`, `min_samples=100`
- Discussed `min_samples` parameter: higher = more conservative, denser cores required

### Codebase Rename: VAEMulti -> VAE
- Renamed `VAEMulti.py` -> `VAE.py`, `VAEMulti.md` -> `VAE.md`
- Updated all file paths and references

### Memory Usage
- `calculate_kmer_frequencies`: Fixed to use temp files + memmap (stays ~1 GB)
- `VAE.py`: Loads all data into memory (~53 GB for 4.8M sequences)

### VAE Training Parameters (optimal config)
- Latent dim: 384, beta: 0.05, LR: 1e-4, batch: 1024, epochs: 1000

## 2026-02-03: Extended training and literature search

### Large Dataset Training
- Combined aquatic (4.8M) + terrestrial (8.0M) + NCBI RefSeq (0.7M) = 13.4M sequences
- Adding 655K NCBI sequences (~5% increase) caused ~30% improvement
- Quality and taxonomic diversity matter more than quantity

### Literature
- Multi-scale k-mer VAE approach appears novel
- Closest: VAMB (4-mers only), GenomeFace (k=1-10 but different architecture)

## 2026-02-05: Per-group CLR + Jeffreys prior

- Switched from joint CLR (mixing 6 compositions) to per-group CLR
- Replaced 1e-6 pseudocount with Jeffreys prior (0.5/n_features)
- Full codebase review identified 10 issues (most now fixed)

## 2026-02-08 to 2026-02-14: Training runs and cross-comparison

- 5 augmented runs (1K-5K bp thresholds) + 5 SFE_SE runs
- Full 5x5 cross-comparison on shuffled data
- Run_3 best augmented (0.702), SFE_SE_5 best marine (0.847)
- SFE_SE models dominate all augmented models
- Data shuffling bug found and fixed (was causing 34-48 pt train/val gap)
- Euclidean > cosine confirmed (+0.076 Spearman)
- Run_5 extreme-GC anomaly identified

## 2026-02-14 to 2026-02-17: Clustering exploration (10 kbp)

- Connected components at d=10: giant component of 1.3M (77% of clustered)
- Hub analysis: top hub in-degree 58,377
- Three graph methods compared: symmetric, mutual, capped
- Leiden sweep across 17 distance thresholds (d=4 to d=12)
- MCL introduced: dramatic improvement over Leiden (GC 6-9 pp vs 40-50 pp)
- MCL cluster size distributions analyzed (power-law-like, median=2)
- Weight function comparison: 1/(d+0.1) >> exp(-d)

## 2026-02-17 to 2026-02-18: 100 kbp analysis

- 154,040 sequences, d-hat=3.74, 78% connected at d=5
- Leiden works reasonably well (7-15 pp GC spans)
- MCL at d=5: excellent (4-6 pp GC at I=3.0)
- d=7 experiment: worse than d=5 at I=1.4, comparable at I=3.0+
- exp(-d) weights: counterintuitively worse
- Confirmed d=5 as best tested distance threshold

## 2026-02-18: nn1 length-threshold sweep

- Computed nn1 for 10K queries at each threshold against ALL sequences
- Clear knee at 50-100 kbp
- 50 kbp: median nn1 crosses below 5 (the clustering threshold)
- Below 50 kbp: noise-dominated; above 200 kbp: diminishing returns

## 2026-02-19: 50 kbp clustering analysis

- 461,674 sequences, d-hat=4.42, median nn1=5.01
- Leiden at d=7: poor (19-46 pp GC spans)
- MCL at d=7: better (4-13 pp) but inconsistent across communities
- Relative threshold argument (d/median_nn1) did NOT hold
- Absolute edge quality matters
- 100 kbp remains best tested threshold
