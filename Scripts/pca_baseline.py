#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "scipy",
#     "scikit-learn",
# ]
# ///
"""
PCA baseline for comparison with VAE embedding.

Tests whether PCA on CLR-transformed multi-scale k-mer frequencies achieves
comparable local distance preservation (Spearman correlation) to the VAE.

Baselines tested:
1. Raw CLR (2772-dim) - no dimensionality reduction at all
2. PCA at various dimensions (32, 128, 384)

Evaluation protocol matches verify_local_distances.py exactly:
- 100 random queries, 50 nearest neighbors, Euclidean distance
- Spearman correlation between embedding distance and k-mer MSE
- Bootstrap CIs over query-level resampling
"""

import argparse
import time

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA

# Constants matching VAE.py
INPUT_DIM = 2772
COL_START = 1
COL_END = 2773
SEED = 42

KMER_SLICES = [
    (0, 2080),       # 6-mer
    (2080, 2592),    # 5-mer
    (2592, 2728),    # 4-mer
    (2728, 2760),    # 3-mer
    (2760, 2770),    # 2-mer
    (2770, 2772),    # 1-mer
]


def clr_transform(data: np.ndarray) -> np.ndarray:
    """Apply per-group Centered Log-Ratio (CLR) transformation.

    Each k-mer size group is CLR-transformed independently, since
    each group is separately normalized to sum to 1.0. Uses a Jeffreys
    prior pseudocount of 0.5/n_features per group.
    """
    data = data.copy()
    for start, end in KMER_SLICES:
        group = data[:, start:end]
        pseudocount = 0.5 / (end - start)
        group += pseudocount
        np.log(group, out = group)
        log_geom_mean = np.mean(group, axis = 1, keepdims = True)
        group -= log_geom_mean
    return data


def evaluate_embedding(embeddings: np.ndarray, clr_data: np.ndarray,
                       n_queries: int = 100, n_neighbors: int = 50,
                       n_bootstrap: int = 10000, label: str = '') -> dict:
    """Run the standard Spearman evaluation protocol.

    Matches verify_local_distances.py exactly:
    - Random queries with seed=42
    - Euclidean distance in embedding space
    - MSE in CLR input space
    - Spearman correlation over all (distance, MSE) pairs
    - Bootstrap CIs at query level
    """
    sample_size = len(embeddings)
    np.random.seed(42)
    query_indices = np.random.choice(sample_size, n_queries, replace = False)

    all_latent_distances = []
    all_kmer_mses = []

    for q_idx in query_indices:
        # Euclidean distances in embedding space
        query_emb = embeddings[q_idx:q_idx + 1]
        latent_distances = cdist(query_emb, embeddings, metric = 'euclidean')[0]

        # K-mer MSE in CLR input space
        query_kmer = clr_data[q_idx:q_idx + 1]
        kmer_mses = np.mean((clr_data - query_kmer) ** 2, axis = 1)

        # K nearest neighbors (excluding self)
        neighbor_indices = np.argsort(latent_distances)[1:n_neighbors + 1]

        for n_idx in neighbor_indices:
            all_latent_distances.append(latent_distances[n_idx])
            all_kmer_mses.append(kmer_mses[n_idx])

    all_latent_distances = np.array(all_latent_distances)
    all_kmer_mses = np.array(all_kmer_mses)

    spearman_r, spearman_p = spearmanr(all_latent_distances, all_kmer_mses)
    pearson_r, pearson_p = pearsonr(all_latent_distances, all_kmer_mses)

    result = {
        'label': label,
        'dims': embeddings.shape[1],
        'spearman': spearman_r,
        'pearson': pearson_r,
    }

    # Bootstrap CI over queries
    if n_bootstrap > 0:
        latent_by_query = all_latent_distances.reshape(n_queries, n_neighbors)
        kmer_by_query = all_kmer_mses.reshape(n_queries, n_neighbors)

        rng = np.random.default_rng(42)
        boot_spearman = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            idx = rng.choice(n_queries, size = n_queries, replace = True)
            flat_lat = latent_by_query[idx].ravel()
            flat_kmer = kmer_by_query[idx].ravel()
            boot_spearman[b] = spearmanr(flat_lat, flat_kmer).statistic

        sp_lo, sp_hi = np.percentile(boot_spearman, [2.5, 97.5])
        result['ci_lo'] = sp_lo
        result['ci_hi'] = sp_hi

    return result


def print_result(r: dict) -> None:
    """Print a single evaluation result."""
    ci = ''
    if 'ci_lo' in r:
        ci = f' [{r["ci_lo"]:.4f}, {r["ci_hi"]:.4f}]'
    print(f'  {r["label"]:>20s} ({r["dims"]:>4d} dims): '
          f'Spearman = {r["spearman"]:.4f}{ci}  '
          f'Pearson = {r["pearson"]:.4f}')


def main():
    parser = argparse.ArgumentParser(
        description = 'PCA baseline comparison for VAE embedding')
    parser.add_argument('-i', '--input', required = True,
                        help = 'K-mer data for evaluation (.npy)')
    parser.add_argument('--train-input', default = None,
                        help = 'K-mer data for fitting PCA (.npy). '
                               'If omitted, uses --input (in-domain PCA)')
    parser.add_argument('--sample-size', type = int, default = 50000,
                        help = 'Validation sample size (default: 50000)')
    parser.add_argument('--pca-train-size', type = int, default = 500000,
                        help = 'Max sequences for PCA fitting (default: 500000)')
    parser.add_argument('--pca-dims', type = int, nargs = '+',
                        default = [32, 128, 384],
                        help = 'PCA dimensions to test (default: 32 128 384)')
    parser.add_argument('--bootstrap', type = int, default = 10000,
                        help = 'Bootstrap resamples (default: 10000)')
    parser.add_argument('-n', '--num-queries', type = int, default = 100,
                        help = 'Number of queries (default: 100)')
    parser.add_argument('-k', '--neighbors', type = int, default = 50,
                        help = 'Nearest neighbors (default: 50)')
    args = parser.parse_args()

    # --- Load evaluation data (validation split) ---
    print(f'Loading evaluation data from {args.input}...')
    data_mmap = np.load(args.input, mmap_mode = 'r')
    n_total = len(data_mmap)
    val_start = int(n_total * 0.9)
    sample_size = min(args.sample_size, n_total - val_start)

    print(f'  Total sequences: {n_total:,}')
    print(f'  Validation split: {val_start:,} to {n_total:,} '
          f'({n_total - val_start:,} sequences)')
    print(f'  Sample size: {sample_size:,}')

    val_data = data_mmap[val_start:val_start + sample_size,
                         COL_START:COL_END].astype(np.float32)
    clr_val = clr_transform(val_data)
    del val_data

    # --- Load PCA training data ---
    train_file = args.train_input or args.input
    print(f'\nLoading PCA training data from {train_file}...')
    train_mmap = np.load(train_file, mmap_mode = 'r')
    n_train_total = int(len(train_mmap) * 0.9)  # first 90%
    pca_train_size = min(args.pca_train_size, n_train_total)

    # Subsample training data uniformly if needed
    if pca_train_size < n_train_total:
        rng = np.random.default_rng(SEED)
        train_indices = rng.choice(n_train_total, pca_train_size, replace = False)
        train_indices.sort()
        print(f'  Subsampling {pca_train_size:,} from {n_train_total:,} training sequences')
        train_data = train_mmap[train_indices, COL_START:COL_END].astype(np.float32)
    else:
        print(f'  Using all {n_train_total:,} training sequences')
        train_data = train_mmap[:n_train_total, COL_START:COL_END].astype(np.float32)

    clr_train = clr_transform(train_data)
    del train_data

    results = []

    # --- Baseline 1: Raw CLR (no dimensionality reduction) ---
    print(f'\n{"=" * 70}')
    print('Evaluating raw CLR (2772 dims, no dimensionality reduction)...')
    t0 = time.time()
    r = evaluate_embedding(clr_val, clr_val,
                           n_queries = args.num_queries,
                           n_neighbors = args.neighbors,
                           n_bootstrap = args.bootstrap,
                           label = 'Raw CLR')
    print_result(r)
    print(f'  Time: {time.time() - t0:.1f}s')
    results.append(r)

    # --- PCA baselines ---
    max_dim = max(args.pca_dims)
    print(f'\nFitting PCA (up to {max_dim} components) on {len(clr_train):,} sequences...')
    t0 = time.time()
    pca = PCA(n_components = max_dim, random_state = SEED)
    pca.fit(clr_train)
    print(f'  PCA fit time: {time.time() - t0:.1f}s')

    # Report variance explained
    for d in args.pca_dims:
        var_explained = np.sum(pca.explained_variance_ratio_[:d])
        print(f'  {d} PCs explain {var_explained:.1%} of variance')

    # Transform validation data
    val_pca_full = pca.transform(clr_val)

    for d in sorted(args.pca_dims):
        print(f'\nEvaluating PCA-{d}...')
        val_pca = val_pca_full[:, :d]
        t0 = time.time()
        r = evaluate_embedding(val_pca, clr_val,
                               n_queries = args.num_queries,
                               n_neighbors = args.neighbors,
                               n_bootstrap = args.bootstrap,
                               label = f'PCA-{d}')
        print_result(r)
        print(f'  Time: {time.time() - t0:.1f}s')
        results.append(r)

    # --- Summary ---
    print(f'\n{"=" * 70}')
    print('SUMMARY')
    print(f'{"=" * 70}')
    print(f'Evaluation: {sample_size:,} sequences from {args.input}')
    print(f'PCA trained on: {len(clr_train):,} sequences from {train_file}')
    print(f'Protocol: {args.num_queries} queries x {args.neighbors} neighbors, '
          f'Euclidean distance')
    print()
    print(f'{"Method":>20s} {"Dims":>6s} {"Spearman":>10s} {"95% CI":>20s} {"Pearson":>10s}')
    print('-' * 70)
    for r in results:
        ci = ''
        if 'ci_lo' in r:
            ci = f'[{r["ci_lo"]:.4f}, {r["ci_hi"]:.4f}]'
        print(f'{r["label"]:>20s} {r["dims"]:>6d} {r["spearman"]:>10.4f} '
              f'{ci:>20s} {r["pearson"]:>10.4f}')

    # Reference values for context
    print()
    print('Reference (VAE NCBI_5 on this data):')
    print('  Spearman ~ 0.831-0.837 (brackish), 0.934 (NCBI)')


if __name__ == '__main__':
    main()
