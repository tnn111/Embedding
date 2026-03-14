#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "scikit-learn",
# ]
# ///
"""
PCA and raw CLR clustering baselines for comparison with VAE embedding.

Runs the full pipeline: embed → k-NN graph → MCL → GC span analysis.
Compares raw CLR (2772 dims), PCA-384, and existing VAE results.

The distance threshold for graph construction is calibrated per method
to match the VAE graph's connectivity (~87% of sequences connected).
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

# Constants matching VAE.py
COL_START = 1
COL_END = 2773
SEED = 42
MIN_LENGTH = 100_000
K_NEIGHBORS = 50
INDEGREE_CAP = 100
EPSILON = 0.1
MCL_INFLATION = 3.0
MCL_THREADS = 16

KMER_SLICES = [
    (0, 2080),       # 6-mer
    (2080, 2592),    # 5-mer
    (2592, 2728),    # 4-mer
    (2728, 2760),    # 3-mer
    (2760, 2770),    # 2-mer
    (2770, 2772),    # 1-mer
]


def clr_transform(data: np.ndarray) -> np.ndarray:
    """Apply per-group Centered Log-Ratio (CLR) transformation."""
    data = data.copy()
    for start, end in KMER_SLICES:
        group = data[:, start:end]
        pseudocount = 0.5 / (end - start)
        group += pseudocount
        np.log(group, out = group)
        log_geom_mean = np.mean(group, axis = 1, keepdims = True)
        group -= log_geom_mean
    return data


def compute_knn(embeddings: np.ndarray, k: int = 50,
                chunk_size: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Compute k nearest neighbors using chunked matrix multiplication.

    Returns:
        nn_indices: (n, k) array of neighbor indices
        nn_distances: (n, k) array of Euclidean distances
    """
    n, d = embeddings.shape
    print(f'  Computing {k}-NN for {n:,} sequences in {d} dims '
          f'(chunks of {chunk_size})...')

    sq_norms = np.sum(embeddings ** 2, axis = 1)
    nn_indices = np.empty((n, k), dtype = np.int32)
    nn_distances = np.empty((n, k), dtype = np.float32)

    t0 = time.time()
    n_chunks = (n + chunk_size - 1) // chunk_size

    for ci, start in enumerate(range(0, n, chunk_size)):
        end = min(start + chunk_size, n)
        # Squared Euclidean: ||a-b||² = ||a||² + ||b||² - 2*a·b
        dots = embeddings[start:end] @ embeddings.T
        sq_dists = sq_norms[start:end, None] + sq_norms[None, :] - 2 * dots
        np.maximum(sq_dists, 0, out = sq_dists)

        # Set self-distance to infinity
        for i in range(end - start):
            sq_dists[i, start + i] = np.inf

        # Find k nearest (argpartition is O(n) vs O(n log n) for argsort)
        topk = np.argpartition(sq_dists, k, axis = 1)[:, :k]
        # Sort the top-k by distance
        for i in range(end - start):
            topk_dists = sq_dists[i, topk[i]]
            order = np.argsort(topk_dists)
            nn_indices[start + i] = topk[i, order]
            nn_distances[start + i] = np.sqrt(topk_dists[order])

        if (ci + 1) % 10 == 0 or ci == n_chunks - 1:
            elapsed = time.time() - t0
            rate = (ci + 1) / elapsed
            eta = (n_chunks - ci - 1) / rate if rate > 0 else 0
            print(f'    Chunk {ci + 1}/{n_chunks} '
                  f'({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)')

        del dots, sq_dists

    print(f'  k-NN complete in {time.time() - t0:.1f}s')
    return nn_indices, nn_distances


def write_neighbor_tsv(path: str, ids: list[str],
                       nn_indices: np.ndarray,
                       nn_distances: np.ndarray) -> None:
    """Write neighbor TSV in the same format as ChromaDB output."""
    n, k = nn_indices.shape
    print(f'  Writing neighbor TSV to {path}...')
    with open(path, 'w') as f:
        for i in range(n):
            parts = [ids[i]]
            for j in range(k):
                ni = nn_indices[i, j]
                nd = nn_distances[i, j]
                parts.append(f'{ids[ni]}({nd:.4f})')
            f.write('\t'.join(parts) + '\n')


def build_graph_abc(neighbor_tsv: str, output_abc: str,
                    d_threshold: float, ids: list[str]) -> dict:
    """Build in-degree capped graph in ABC format from neighbor TSV.

    Returns stats dict with edge counts and connectivity.
    """
    n = len(ids)
    id_to_idx = {id_: i for i, id_ in enumerate(ids)}

    print(f'  Building graph: d < {d_threshold:.2f}, '
          f'in-degree cap {INDEGREE_CAP}...')

    # Pass 1: collect all directed edges
    edges = []
    with open(neighbor_tsv) as f:
        for line in f:
            fields = line.rstrip('\n').split('\t')
            src = fields[0]
            for field in fields[1:]:
                paren = field.rfind('(')
                dist = float(field[paren + 1:-1])
                if dist >= d_threshold:
                    break  # distance-sorted
                tgt = field[:paren]
                edges.append((src, tgt, dist))

    n_directed = len(edges)

    # Compute in-degree and cap
    in_degree = {}
    for _, tgt, _ in edges:
        in_degree[tgt] = in_degree.get(tgt, 0) + 1

    # Sort edges by distance (keep closest edges for each target)
    edges.sort(key = lambda e: e[2])
    capped_in = {}
    kept_edges = []
    for src, tgt, dist in edges:
        if capped_in.get(tgt, 0) < INDEGREE_CAP:
            kept_edges.append((src, tgt, dist))
            capped_in[tgt] = capped_in.get(tgt, 0) + 1

    n_after_cap = len(kept_edges)

    # Write ABC format (undirected: keep unique pairs)
    seen = set()
    with open(output_abc, 'w') as f:
        for src, tgt, dist in kept_edges:
            pair = tuple(sorted([src, tgt]))
            if pair not in seen:
                seen.add(pair)
                weight = 1.0 / (dist + EPSILON)
                f.write(f'{src}\t{tgt}\t{weight:.6f}\n')

    n_undirected = len(seen)

    # Connectivity
    connected = set()
    for src, tgt, _ in kept_edges:
        connected.add(src)
        connected.add(tgt)
    n_connected = len(connected)
    pct_connected = 100 * n_connected / n

    stats = {
        'directed_edges': n_directed,
        'after_cap': n_after_cap,
        'undirected_edges': n_undirected,
        'connected': n_connected,
        'pct_connected': pct_connected,
    }

    print(f'    Directed edges: {n_directed:,}')
    print(f'    After in-degree cap: {n_after_cap:,} '
          f'(dropped {n_directed - n_after_cap:,})')
    print(f'    Undirected edges: {n_undirected:,}')
    print(f'    Connected sequences: {n_connected:,} ({pct_connected:.1f}%)')

    return stats


def run_mcl(abc_path: str, output_dir: str, inflation: float = 3.0) -> str:
    """Run MCL on ABC format graph. Returns path to cluster file."""
    os.makedirs(output_dir, exist_ok = True)
    tab_file = os.path.join(output_dir, 'graph.tab')
    mci_file = os.path.join(output_dir, 'graph.mci')
    cluster_file = os.path.join(output_dir, f'mcl_I{inflation}.clusters')

    # Convert ABC to binary
    if not os.path.exists(mci_file):
        print(f'  Converting ABC to MCL binary format...')
        subprocess.run([
            'mcxload', '-abc', abc_path,
            '--stream-mirror',
            '-write-tab', tab_file,
            '-o', mci_file
        ], check = True, capture_output = True)

    # Run MCL
    print(f'  Running MCL with I={inflation}, {MCL_THREADS} threads...')
    t0 = time.time()
    subprocess.run([
        'mcl', mci_file,
        '-I', str(inflation),
        '-te', str(MCL_THREADS),
        '-o', cluster_file,
        '-use-tab', tab_file
    ], check = True, capture_output = True)
    print(f'  MCL complete in {time.time() - t0:.1f}s')

    return cluster_file


def analyze_clusters(cluster_file: str, ids: list[str],
                     gc_content: np.ndarray, lengths: np.ndarray,
                     n_top: int = 10) -> dict:
    """Load MCL clusters and compute GC span statistics."""
    id_to_idx = {id_: i for i, id_ in enumerate(ids)}

    clusters = []
    with open(cluster_file) as f:
        for line in f:
            members = line.strip().split('\t')
            indices = [id_to_idx[m] for m in members if m in id_to_idx]
            if len(indices) >= 2:
                clusters.append(indices)

    clusters.sort(key = len, reverse = True)
    n_total = len(clusters)
    n_singletons_file = sum(1 for line in open(cluster_file)
                            if '\t' not in line.strip())
    n_clustered = sum(len(c) for c in clusters)

    print(f'\n  Clusters: {n_total:,} non-singleton, '
          f'{n_singletons_file:,} singletons')
    print(f'  Clustered sequences: {n_clustered:,}')

    # GC spans for top clusters
    print(f'\n  Top {n_top} clusters by size:')
    print(f'  {"Rank":>4s}  {"Size":>6s}  {"GC span (pp)":>12s}  '
          f'{"GC mean":>8s}  {"Len median":>10s}')
    print(f'  {"-" * 50}')

    gc_spans = []
    for rank, indices in enumerate(clusters[:n_top]):
        gc = gc_content[indices]
        gc_span = 100 * (gc.max() - gc.min())
        gc_mean = gc.mean()
        med_len = np.median(lengths[indices])
        gc_spans.append(gc_span)
        print(f'  {rank + 1:>4d}  {len(indices):>6,}  {gc_span:>12.1f}  '
              f'{gc_mean:>8.3f}  {med_len:>10,.0f}')

    stats = {
        'n_clusters': n_total,
        'n_singletons': n_singletons_file,
        'n_clustered': n_clustered,
        'gc_spans_top3': gc_spans[:3] if len(gc_spans) >= 3 else gc_spans,
        'gc_spans_top10': gc_spans,
    }
    return stats


def find_threshold_for_connectivity(nn_distances: np.ndarray,
                                    target_pct: float) -> float:
    """Find distance threshold that connects target_pct of sequences."""
    nn1 = nn_distances[:, 0]  # nearest neighbor distance
    # Find threshold where target_pct of sequences have nn1 < threshold
    threshold = np.percentile(nn1, target_pct)
    return threshold


def main():
    parser = argparse.ArgumentParser(
        description = 'PCA/CLR clustering baseline comparison')
    parser.add_argument('--kmers', default = 'Runs/kmers_SFE_SE_1.npy',
                        help = 'K-mer data file (default: Runs/kmers_SFE_SE_1.npy)')
    parser.add_argument('--ids', default = 'Runs/ids_SFE_SE_1.txt',
                        help = 'Sequence IDs file')
    parser.add_argument('--pca-train', default = 'Runs/kmers_NCBI_5.npy',
                        help = 'K-mer data for fitting PCA')
    parser.add_argument('--pca-train-size', type = int, default = 500_000,
                        help = 'Max sequences for PCA fitting')
    parser.add_argument('--vae-clusters',
                        default = 'Runs/MCL_100_NCBI_5_d5/mcl_I3.0.clusters',
                        help = 'Existing VAE MCL cluster file for comparison')
    parser.add_argument('--output-dir', default = 'Runs/PCA_baseline',
                        help = 'Output directory')
    parser.add_argument('--target-connectivity', type = float, default = 86.8,
                        help = 'Target connectivity %% for threshold calibration')
    parser.add_argument('--chunk-size', type = int, default = 2000,
                        help = 'Chunk size for k-NN computation')
    parser.add_argument('--skip-clr', action = 'store_true',
                        help = 'Skip raw CLR baseline (slow at 2772 dims)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok = True)

    # ================================================================
    # Step 1: Load data and filter to >= 100 kbp
    # ================================================================
    print('=' * 70)
    print('STEP 1: Loading data')
    print('=' * 70)

    print(f'Loading k-mer data from {args.kmers}...')
    kmers_mmap = np.load(args.kmers, mmap_mode = 'r')
    n_total = len(kmers_mmap)
    print(f'  Total sequences: {n_total:,}')

    # Filter by length (column 0)
    lengths_all = kmers_mmap[:, 0]
    long_mask = np.array(lengths_all >= MIN_LENGTH)
    n_long = long_mask.sum()
    long_indices = np.where(long_mask)[0]
    print(f'  Sequences >= {MIN_LENGTH:,} bp: {n_long:,} ({100 * n_long / n_total:.1f}%)')

    # Load filtered data
    print('  Loading filtered k-mer data into memory...')
    t0 = time.time()
    kmers = kmers_mmap[long_indices].astype(np.float32)
    print(f'  Loaded in {time.time() - t0:.1f}s, shape: {kmers.shape}')

    lengths = kmers[:, 0]
    kmer_features = kmers[:, COL_START:COL_END]

    # GC content = second 1-mer feature (column 2772, 0-indexed in kmer_features: 2771)
    gc_content = kmer_features[:, 2771]

    # Load IDs
    print(f'Loading IDs from {args.ids}...')
    with open(args.ids) as f:
        all_ids = [line.strip() for line in f]
    ids = [all_ids[i] for i in long_indices]
    print(f'  Loaded {len(ids):,} IDs')

    # CLR transform
    print('Applying CLR transformation...')
    t0 = time.time()
    clr_data = clr_transform(kmer_features)
    print(f'  CLR transform complete in {time.time() - t0:.1f}s')

    del kmers_mmap, kmer_features, kmers

    # ================================================================
    # Step 2: Fit PCA on NCBI_5 training data
    # ================================================================
    print(f'\n{"=" * 70}')
    print('STEP 2: Fitting PCA on NCBI_5 training data')
    print('=' * 70)

    print(f'Loading PCA training data from {args.pca_train}...')
    train_mmap = np.load(args.pca_train, mmap_mode = 'r')
    n_train_total = int(len(train_mmap) * 0.9)  # first 90% = training split
    pca_train_size = min(args.pca_train_size, n_train_total)

    rng = np.random.default_rng(SEED)
    if pca_train_size < n_train_total:
        train_indices = rng.choice(n_train_total, pca_train_size, replace = False)
        train_indices.sort()
        print(f'  Subsampling {pca_train_size:,} from {n_train_total:,}')
        train_data = train_mmap[train_indices, COL_START:COL_END].astype(np.float32)
    else:
        print(f'  Using all {n_train_total:,} training sequences')
        train_data = train_mmap[:n_train_total, COL_START:COL_END].astype(np.float32)

    clr_train = clr_transform(train_data)
    del train_data, train_mmap

    print('  Fitting PCA (384 components)...')
    t0 = time.time()
    pca = PCA(n_components = 384, random_state = SEED)
    pca.fit(clr_train)
    var_explained = np.sum(pca.explained_variance_ratio_)
    print(f'  PCA fit in {time.time() - t0:.1f}s, '
          f'384 PCs explain {var_explained:.1%} of variance')
    del clr_train

    # Project the 100 kbp data
    print('  Projecting 100 kbp data...')
    pca_embeddings = pca.transform(clr_data).astype(np.float32)
    print(f'  PCA embeddings shape: {pca_embeddings.shape}')

    # ================================================================
    # Step 3: Define methods to evaluate
    # ================================================================
    methods = []

    methods.append({
        'name': 'PCA-384',
        'embeddings': pca_embeddings,
        'dims': 384,
    })

    if not args.skip_clr:
        methods.append({
            'name': 'CLR-2772',
            'embeddings': clr_data,
            'dims': 2772,
        })

    # ================================================================
    # Step 4: Run pipeline for each method
    # ================================================================
    all_results = {}

    for method in methods:
        name = method['name']
        emb = method['embeddings']

        print(f'\n{"=" * 70}')
        print(f'METHOD: {name} ({emb.shape[1]} dims)')
        print('=' * 70)

        method_dir = os.path.join(args.output_dir, name.replace('-', '_'))
        os.makedirs(method_dir, exist_ok = True)

        # --- k-NN search ---
        neighbor_tsv = os.path.join(method_dir, 'neighbors.tsv')
        nn_cache = os.path.join(method_dir, 'nn_cache.npz')

        if os.path.exists(nn_cache):
            print(f'  Loading cached k-NN from {nn_cache}...')
            cache = np.load(nn_cache)
            nn_indices = cache['indices']
            nn_distances = cache['distances']
        else:
            nn_indices, nn_distances = compute_knn(
                emb, k = K_NEIGHBORS, chunk_size = args.chunk_size)
            np.savez(nn_cache, indices = nn_indices, distances = nn_distances)

        # Write neighbor TSV
        if not os.path.exists(neighbor_tsv):
            write_neighbor_tsv(neighbor_tsv, ids, nn_indices, nn_distances)

        # --- Distance statistics ---
        nn1 = nn_distances[:, 0]
        print(f'\n  nn1 distance statistics:')
        for p in [50, 75, 87, 90, 95, 99]:
            print(f'    P{p}: {np.percentile(nn1, p):.4f}')

        # --- Calibrate threshold for target connectivity ---
        d_threshold = find_threshold_for_connectivity(
            nn_distances, args.target_connectivity)
        actual_pct = 100 * np.mean(nn1 < d_threshold)
        print(f'\n  Calibrated threshold for {args.target_connectivity}% '
              f'connectivity: d = {d_threshold:.4f}')
        print(f'  Actual connectivity at this threshold: {actual_pct:.1f}%')

        # --- Build graph ---
        abc_path = os.path.join(method_dir, 'graph.tsv')
        graph_stats = build_graph_abc(
            neighbor_tsv, abc_path, d_threshold, ids)

        # --- Run MCL ---
        cluster_file = run_mcl(abc_path, method_dir, MCL_INFLATION)

        # --- Analyze clusters ---
        cluster_stats = analyze_clusters(
            cluster_file, ids, gc_content, lengths, n_top = 10)

        all_results[name] = {
            'threshold': d_threshold,
            'graph': graph_stats,
            'clusters': cluster_stats,
        }

    # ================================================================
    # Step 5: Load and analyze existing VAE clusters for comparison
    # ================================================================
    if os.path.exists(args.vae_clusters):
        print(f'\n{"=" * 70}')
        print('METHOD: VAE-384 (existing results)')
        print('=' * 70)

        cluster_stats = analyze_clusters(
            args.vae_clusters, ids, gc_content, lengths, n_top = 10)
        all_results['VAE-384'] = {
            'threshold': 5.0,
            'clusters': cluster_stats,
        }

    # ================================================================
    # Summary
    # ================================================================
    print(f'\n{"=" * 70}')
    print('COMPARISON SUMMARY')
    print('=' * 70)
    print(f'\nTarget connectivity: {args.target_connectivity}%')
    print(f'MCL inflation: {MCL_INFLATION}')
    print()

    header = (f'{"Method":>12s}  {"Dims":>5s}  {"Threshold":>9s}  '
              f'{"Clusters":>8s}  {"Singletons":>10s}  {"Clustered":>9s}  '
              f'{"GC top3 (pp)":>14s}')
    print(header)
    print('-' * len(header))

    for name in ['VAE-384', 'PCA-384', 'CLR-2772']:
        if name not in all_results:
            continue
        r = all_results[name]
        cs = r['clusters']
        dims = name.split('-')[1]
        threshold = f'{r["threshold"]:.2f}' if 'threshold' in r else 'N/A'
        gc3 = cs.get('gc_spans_top3', [])
        gc3_str = '/'.join(f'{g:.1f}' for g in gc3)
        print(f'{name:>12s}  {dims:>5s}  {threshold:>9s}  '
              f'{cs["n_clusters"]:>8,}  {cs["n_singletons"]:>10,}  '
              f'{cs["n_clustered"]:>9,}  {gc3_str:>14s}')

    print()
    print('GC span comparison (lower = better, more compositionally coherent):')
    for name in ['VAE-384', 'PCA-384', 'CLR-2772']:
        if name not in all_results:
            continue
        gc = all_results[name]['clusters'].get('gc_spans_top10', [])
        if gc:
            print(f'  {name:>12s}: median={np.median(gc):.1f} pp, '
                  f'mean={np.mean(gc):.1f} pp')

    # Save summary
    summary_path = os.path.join(args.output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write('PCA/CLR Clustering Baseline Comparison\n')
        f.write(f'Target connectivity: {args.target_connectivity}%\n')
        f.write(f'MCL inflation: {MCL_INFLATION}\n\n')
        for name, r in all_results.items():
            cs = r['clusters']
            gc3 = cs.get('gc_spans_top3', [])
            f.write(f'{name}:\n')
            f.write(f'  Threshold: {r.get("threshold", "N/A")}\n')
            f.write(f'  Clusters: {cs["n_clusters"]}\n')
            f.write(f'  Singletons: {cs["n_singletons"]}\n')
            f.write(f'  Clustered: {cs["n_clustered"]}\n')
            f.write(f'  GC spans top 3: {gc3}\n')
            f.write(f'  GC spans top 10: {cs.get("gc_spans_top10", [])}\n\n')

    print(f'\nSummary saved to {summary_path}')


if __name__ == '__main__':
    main()
