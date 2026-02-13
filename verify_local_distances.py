#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "keras",
#     "scipy",
#     "jax[cuda12]",
# ]
# ///
"""
Verify that latent space distances correlate with k-mer similarity.

Tests whether "close in latent space" = "similar k-mer profiles" by:
1. Taking random sequences from validation set
2. Finding K nearest neighbors in latent space
3. Computing actual k-mer MSE between query and neighbors
4. Checking correlation between latent distance and k-mer similarity
"""

import os
os.environ['KERAS_BACKEND'] = 'jax'

import argparse
import numpy as np
import keras
from keras import layers, ops
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, pearsonr

# Constants matching VAE.py
INPUT_DIM = 2772
COL_START = 1
COL_END = 2773
SEED = 42


@keras.saving.register_keras_serializable()
class Sampling(layers.Layer):
    """Reparameterization trick for VAE."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(SEED)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape = (batch, dim), seed = self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon

    def get_config(self):
        return super().get_config()


@keras.saving.register_keras_serializable()
class ClipLayer(layers.Layer):
    """Clips tensor values to a specified range."""
    def __init__(self, min_value = -20, max_value = 2, **kwargs):
        super().__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value

    def call(self, inputs):
        return ops.clip(inputs, self.min_value, self.max_value)

    def get_config(self):
        config = super().get_config()
        config.update({'min_value': self.min_value, 'max_value': self.max_value})
        return config


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


def load_data(file_path: str, start_idx: int, end_idx: int) -> np.ndarray:
    """Load k-mer data with CLR transform."""
    data_mmap = np.load(file_path, mmap_mode = 'r')
    data = data_mmap[start_idx:end_idx, COL_START:COL_END].astype(np.float32)
    return clr_transform(data)


def main():
    parser = argparse.ArgumentParser(description = 'Verify latent space local distances')
    parser.add_argument('-i', '--input', required = True, help = 'Path to input .npy file')
    parser.add_argument('-e', '--encoder', default = 'vae_encoder_final.keras',
                        help = 'Path to encoder model')
    parser.add_argument('-n', '--num-queries', type = int, default = 100,
                        help = 'Number of query sequences to test')
    parser.add_argument('-k', '--neighbors', type = int, default = 50,
                        help = 'Number of nearest neighbors to find')
    parser.add_argument('--sample-size', type = int, default = 10000,
                        help = 'Size of sample to search within')
    parser.add_argument('--metric', choices = ['euclidean', 'cosine'], default = 'euclidean',
                        help = 'Distance metric for latent space (default: euclidean)')
    args = parser.parse_args()

    print(f'Loading encoder from {args.encoder}...')
    encoder = keras.models.load_model(args.encoder, custom_objects = {
        'Sampling': Sampling,
        'ClipLayer': ClipLayer
    })

    # Load a sample of data
    print(f'Loading {args.sample_size} samples...')
    data_mmap = np.load(args.input, mmap_mode = 'r')
    n_total = len(data_mmap)

    # Use last 10% as validation (matching VAE.py)
    val_start = int(n_total * 0.9)
    sample_size = min(args.sample_size, n_total - val_start)
    data = load_data(args.input, val_start, val_start + sample_size)

    print(f'Encoding {sample_size} samples...')
    # Get latent representations (z_mean)
    z_mean, _, _ = encoder.predict(data, batch_size = 256, verbose = 0)

    print(f'Computing pairwise distances for {args.num_queries} queries (metric: {args.metric})...')

    # Select random queries
    np.random.seed(42)
    query_indices = np.random.choice(sample_size, args.num_queries, replace = False)

    all_latent_distances = []
    all_kmer_mses = []
    all_ranks = []

    for i, q_idx in enumerate(query_indices):
        # Compute latent distances from query to all others
        query_latent = z_mean[q_idx:q_idx + 1]
        latent_distances = cdist(query_latent, z_mean, metric = args.metric)[0]

        # Compute k-mer MSE from query to all others
        query_kmer = data[q_idx:q_idx + 1]
        kmer_mses = np.mean((data - query_kmer) ** 2, axis = 1)

        # Get K nearest neighbors (excluding self)
        neighbor_indices = np.argsort(latent_distances)[1:args.neighbors + 1]

        for rank, n_idx in enumerate(neighbor_indices):
            all_latent_distances.append(latent_distances[n_idx])
            all_kmer_mses.append(kmer_mses[n_idx])
            all_ranks.append(rank + 1)

    all_latent_distances = np.array(all_latent_distances)
    all_kmer_mses = np.array(all_kmer_mses)
    all_ranks = np.array(all_ranks)

    # Compute correlations
    pearson_r, pearson_p = pearsonr(all_latent_distances, all_kmer_mses)
    spearman_r, spearman_p = spearmanr(all_latent_distances, all_kmer_mses)

    print('\n' + '=' * 60)
    print('RESULTS: Latent Distance vs K-mer MSE Correlation')
    print('=' * 60)
    print(f'Number of queries: {args.num_queries}')
    print(f'Neighbors per query: {args.neighbors}')
    print(f'Total pairs analyzed: {len(all_latent_distances)}')
    print()
    print(f'Pearson correlation:  r = {pearson_r:.4f} (p = {pearson_p:.2e})')
    print(f'Spearman correlation: r = {spearman_r:.4f} (p = {spearman_p:.2e})')
    print()

    # Analyze by neighbor rank
    print('K-mer MSE by neighbor rank (latent space):')
    print('-' * 40)
    for k in [1, 5, 10, 20, 50]:
        if k <= args.neighbors:
            mask = all_ranks <= k
            mean_mse = np.mean(all_kmer_mses[mask])
            std_mse = np.std(all_kmer_mses[mask])
            print(f'  Top {k:2d} neighbors: MSE = {mean_mse:.4f} ± {std_mse:.4f}')

    # Compare to random baseline
    print()
    print('Random baseline (for comparison):')
    random_pairs = np.random.choice(sample_size, size = (args.num_queries * args.neighbors, 2))
    random_mses = np.mean((data[random_pairs[:, 0]] - data[random_pairs[:, 1]]) ** 2, axis = 1)
    print(f'  Random pairs MSE: {np.mean(random_mses):.4f} ± {np.std(random_mses):.4f}')

    print()
    print('Interpretation:')
    if spearman_r > 0.7:
        print('  ✓ STRONG correlation - latent distances reliably predict k-mer similarity')
    elif spearman_r > 0.5:
        print('  ✓ MODERATE correlation - latent distances are meaningful for retrieval')
    elif spearman_r > 0.3:
        print('  ~ WEAK correlation - latent distances somewhat reflect k-mer similarity')
    else:
        print('  ✗ POOR correlation - latent distances may not be reliable for retrieval')


if __name__ == '__main__':
    main()
