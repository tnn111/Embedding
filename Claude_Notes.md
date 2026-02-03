# Claude Session Notes

## About This File

Torben collaborates with multiple Claude instances:
- **This host (Threadripper)**: Heavy-duty computation - VAE training, large dataset processing, clustering
- **Desktop**: Writing and lighter tasks

These notes are shared across both so each instance can understand context from the other's sessions. Keep notes clear and comprehensive.

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

Training on larger combined dataset (aquatic + terrestrial) shows interesting trade-offs:

| Metric | Baseline (4.8M aquatic) | New (larger mixed) | Change |
|--------|-------------------------|---------------------|--------|
| Val loss | 2899 | 1953 | -33% ✓ |
| MSE | 1.030 | 0.899 | -13% ✓ |
| 6-mer MSE | 1.319 | 1.146 | -13% ✓ |
| 5-mer MSE | 0.203 | 0.191 | -6% ✓ |
| 4-mer MSE | 0.053 | 0.064 | +20% ✗ |
| 3-mer MSE | 0.015 | 0.020 | +33% ✗ |
| 2-mer MSE | 0.008 | 0.009 | +12% ✗ |
| 1-mer MSE | 0.005 | 0.006 | +20% ✗ |

**Key observation**: Model trades shorter k-mer accuracy for better 6-mer reconstruction. The increased diversity forces latent space to prioritize higher-dimensional 6-mer signal (2080 features) at expense of lower k-mers (32 features for 3-mers).

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
