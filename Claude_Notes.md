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
- Running `calculate_kmer_frequencies` on a larger dataset (several times bigger)
- Monitoring memory usage to see if optimization is needed
