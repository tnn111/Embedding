#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
# ]
# ///
"""
Edge-matched clustering baseline: VAE at tighter thresholds vs PCA-384.

The v1 comparison matched nn1 connectivity but resulted in 60% more edges
for VAE (3.55M vs 2.21M). This script re-runs the VAE clustering at
tighter thresholds to match PCA's edge count for a fair comparison.

Reuses existing neighbor TSVs — only rebuilds graphs and re-runs MCL.
"""

import os
import subprocess
import time

import numpy as np

# Constants
INDEGREE_CAP = 100
EPSILON = 0.1
MCL_INFLATION = 3.0
MCL_THREADS = 16
MIN_LENGTH = 100_000
COL_START = 1
COL_END = 2773

KMER_SLICES = [
    (0, 2080),       # 6-mer
    (2080, 2592),    # 5-mer
    (2592, 2728),    # 4-mer
    (2728, 2760),    # 3-mer
    (2760, 2770),    # 2-mer
    (2770, 2772),    # 1-mer
]


def build_graph_abc(neighbor_tsv: str, output_abc: str,
                    d_threshold: float, ids: list[str]) -> dict:
    """Build in-degree capped graph in ABC format from neighbor TSV."""
    n = len(ids)
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

    print(f'    Directed edges: {n_directed:,}')
    print(f'    After in-degree cap: {n_after_cap:,}')
    print(f'    Undirected edges: {n_undirected:,}')
    print(f'    Connected: {n_connected:,} ({pct_connected:.1f}%)')

    return {
        'directed_edges': n_directed,
        'after_cap': n_after_cap,
        'undirected_edges': n_undirected,
        'connected': n_connected,
        'pct_connected': pct_connected,
    }


def run_mcl(abc_path: str, output_dir: str, inflation: float = 3.0) -> str:
    """Run MCL on ABC format graph."""
    os.makedirs(output_dir, exist_ok = True)
    tab_file = os.path.join(output_dir, 'graph.tab')
    mci_file = os.path.join(output_dir, 'graph.mci')
    cluster_file = os.path.join(output_dir, f'mcl_I{inflation}.clusters')

    print(f'  Converting ABC to MCL binary format...')
    subprocess.run([
        'mcxload', '-abc', abc_path,
        '--stream-mirror',
        '-write-tab', tab_file,
        '-o', mci_file
    ], check = True, capture_output = True)

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
    singletons = 0
    with open(cluster_file) as f:
        for line in f:
            members = line.strip().split('\t')
            indices = [id_to_idx[m] for m in members if m in id_to_idx]
            if len(indices) >= 2:
                clusters.append(indices)
            elif len(indices) == 1:
                singletons += 1

    clusters.sort(key = len, reverse = True)
    n_clustered = sum(len(c) for c in clusters)

    # GC spans for ALL clusters
    all_gc_spans = []
    all_sizes = []
    for indices in clusters:
        gc = gc_content[indices]
        span = 100 * (gc.max() - gc.min())
        all_gc_spans.append(span)
        all_sizes.append(len(indices))

    all_gc_spans = np.array(all_gc_spans)
    all_sizes = np.array(all_sizes)

    # Print top clusters
    print(f'\n  Clusters: {len(clusters):,} non-singleton, '
          f'{singletons:,} singletons')
    print(f'  Clustered sequences: {n_clustered:,}')
    print(f'\n  Top {n_top} clusters by size:')
    print(f'  {"Rank":>4s}  {"Size":>6s}  {"GC span (pp)":>12s}  '
          f'{"GC mean":>8s}  {"Len median":>10s}')
    print(f'  {"-" * 50}')

    for rank, indices in enumerate(clusters[:n_top]):
        gc = gc_content[indices]
        gc_span = 100 * (gc.max() - gc.min())
        gc_mean = gc.mean()
        med_len = np.median(lengths[indices])
        print(f'  {rank + 1:>4d}  {len(indices):>6,}  {gc_span:>12.1f}  '
              f'{gc_mean:>8.3f}  {med_len:>10,.0f}')

    # Summary statistics
    weighted_mean = np.average(all_gc_spans, weights = all_sizes)

    print(f'\n  GC span summary (all {len(clusters):,} clusters):')
    print(f'    Median: {np.median(all_gc_spans):.2f} pp')
    print(f'    Mean: {np.mean(all_gc_spans):.2f} pp')
    print(f'    Weighted mean: {weighted_mean:.2f} pp')
    print(f'    P75: {np.percentile(all_gc_spans, 75):.2f} pp')
    print(f'    P90: {np.percentile(all_gc_spans, 90):.2f} pp')
    print(f'    < 5 pp: {np.sum(all_gc_spans < 5):,} '
          f'({100 * np.mean(all_gc_spans < 5):.1f}%)')

    return {
        'n_clusters': len(clusters),
        'n_singletons': singletons,
        'n_clustered': n_clustered,
        'gc_spans_all': all_gc_spans,
        'sizes_all': all_sizes,
        'gc_median': np.median(all_gc_spans),
        'gc_mean': np.mean(all_gc_spans),
        'gc_weighted_mean': weighted_mean,
        'gc_p75': np.percentile(all_gc_spans, 75),
        'gc_p90': np.percentile(all_gc_spans, 90),
        'pct_lt5': 100 * np.mean(all_gc_spans < 5),
    }


def main():
    # ================================================================
    # Load shared data
    # ================================================================
    print('=' * 70)
    print('Loading data...')
    print('=' * 70)

    kmers_mmap = np.load('Runs/kmers_SFE_SE_1.npy', mmap_mode = 'r')
    lengths_all = kmers_mmap[:, 0]
    long_mask = np.array(lengths_all >= MIN_LENGTH)
    long_indices = np.where(long_mask)[0]
    n_long = len(long_indices)
    print(f'Sequences >= {MIN_LENGTH:,} bp: {n_long:,}')

    lengths = kmers_mmap[long_indices, 0].astype(np.float32)
    gc_content = kmers_mmap[long_indices, 2772].astype(np.float32)

    with open('Runs/ids_SFE_SE_1.txt') as f:
        all_ids = [l.strip() for l in f]
    ids = [all_ids[i] for i in long_indices]
    print(f'Loaded {len(ids):,} IDs')

    # ================================================================
    # PCA-384 reference (from v1 run)
    # ================================================================
    print(f'\n{"=" * 70}')
    print('PCA-384 reference (from v1 run)')
    print('=' * 70)

    pca_stats = analyze_clusters(
        'Runs/PCA_baseline/PCA_384/mcl_I3.0.clusters',
        ids, gc_content, lengths)
    pca_edges = 2_212_069  # from v1 run

    # ================================================================
    # VAE at original d=5.0 (reference)
    # ================================================================
    print(f'\n{"=" * 70}')
    print('VAE-384 at d=5.0 (original)')
    print('=' * 70)

    vae50_stats = analyze_clusters(
        'Runs/MCL_100_NCBI_5_d5/mcl_I3.0.clusters',
        ids, gc_content, lengths)

    # ================================================================
    # VAE at tighter thresholds — target ~2.2M undirected edges
    # ================================================================
    vae_neighbor_tsv = 'Runs/neighbors_100_NCBI_5.tsv'
    output_base = 'Runs/PCA_baseline'

    # Try a range of thresholds
    thresholds = [3.5, 3.8, 4.0, 4.2]
    vae_results = {}

    for d_thresh in thresholds:
        label = f'VAE_d{d_thresh:.1f}'
        print(f'\n{"=" * 70}')
        print(f'VAE-384 at d={d_thresh:.1f}')
        print('=' * 70)

        method_dir = os.path.join(output_base, label)
        abc_path = os.path.join(method_dir, 'graph.tsv')
        os.makedirs(method_dir, exist_ok = True)

        # Build graph
        graph_stats = build_graph_abc(
            vae_neighbor_tsv, abc_path, d_thresh, ids)

        # Run MCL
        cluster_file = run_mcl(abc_path, method_dir, MCL_INFLATION)

        # Analyze
        cluster_stats = analyze_clusters(
            cluster_file, ids, gc_content, lengths)

        vae_results[d_thresh] = {
            'graph': graph_stats,
            'clusters': cluster_stats,
        }

    # ================================================================
    # Summary comparison
    # ================================================================
    print(f'\n{"=" * 70}')
    print('EDGE-MATCHED COMPARISON SUMMARY')
    print('=' * 70)

    header = (f'{"Method":>14s}  {"Threshold":>9s}  {"Edges":>10s}  '
              f'{"Clusters":>8s}  {"Singletons":>10s}  '
              f'{"GC med":>7s}  {"GC mean":>7s}  {"GC wmean":>8s}  '
              f'{"GC P90":>7s}  {"<5pp":>6s}')
    print(header)
    print('-' * len(header))

    # PCA reference
    print(f'{"PCA-384":>14s}  {"4.62":>9s}  {pca_edges:>10,}  '
          f'{pca_stats["n_clusters"]:>8,}  {pca_stats["n_singletons"]:>10,}  '
          f'{pca_stats["gc_median"]:>7.2f}  {pca_stats["gc_mean"]:>7.2f}  '
          f'{pca_stats["gc_weighted_mean"]:>8.2f}  '
          f'{pca_stats["gc_p90"]:>7.2f}  '
          f'{pca_stats["pct_lt5"]:>5.1f}%')

    # VAE at d=5.0
    print(f'{"VAE d=5.0":>14s}  {"5.00":>9s}  {3_550_516:>10,}  '
          f'{vae50_stats["n_clusters"]:>8,}  {vae50_stats["n_singletons"]:>10,}  '
          f'{vae50_stats["gc_median"]:>7.2f}  {vae50_stats["gc_mean"]:>7.2f}  '
          f'{vae50_stats["gc_weighted_mean"]:>8.2f}  '
          f'{vae50_stats["gc_p90"]:>7.2f}  '
          f'{vae50_stats["pct_lt5"]:>5.1f}%')

    # VAE at tighter thresholds
    for d_thresh in thresholds:
        r = vae_results[d_thresh]
        gs = r['graph']
        cs = r['clusters']
        label = f'VAE d={d_thresh:.1f}'
        edges = gs['undirected_edges']
        match = ' <-- edge-matched' if abs(edges - pca_edges) / pca_edges < 0.1 else ''
        print(f'{label:>14s}  {d_thresh:>9.2f}  {edges:>10,}  '
              f'{cs["n_clusters"]:>8,}  {cs["n_singletons"]:>10,}  '
              f'{cs["gc_median"]:>7.2f}  {cs["gc_mean"]:>7.2f}  '
              f'{cs["gc_weighted_mean"]:>8.2f}  '
              f'{cs["gc_p90"]:>7.2f}  '
              f'{cs["pct_lt5"]:>5.1f}%{match}')

    # Save summary
    summary_path = os.path.join(output_base, 'summary_v2.txt')
    with open(summary_path, 'w') as f:
        f.write('Edge-Matched Clustering Baseline Comparison (v2)\n')
        f.write(f'MCL inflation: {MCL_INFLATION}\n')
        f.write(f'In-degree cap: {INDEGREE_CAP}\n\n')

        f.write(f'PCA-384 (reference):\n')
        f.write(f'  Threshold: 4.62\n')
        f.write(f'  Edges: {pca_edges:,}\n')
        for k in ['n_clusters', 'n_singletons', 'n_clustered',
                   'gc_median', 'gc_mean', 'gc_weighted_mean',
                   'gc_p75', 'gc_p90', 'pct_lt5']:
            f.write(f'  {k}: {pca_stats[k]}\n')

        f.write(f'\nVAE d=5.0 (original):\n')
        f.write(f'  Threshold: 5.00\n')
        f.write(f'  Edges: 3,550,516\n')
        for k in ['n_clusters', 'n_singletons', 'n_clustered',
                   'gc_median', 'gc_mean', 'gc_weighted_mean',
                   'gc_p75', 'gc_p90', 'pct_lt5']:
            f.write(f'  {k}: {vae50_stats[k]}\n')

        for d_thresh in thresholds:
            r = vae_results[d_thresh]
            gs = r['graph']
            cs = r['clusters']
            f.write(f'\nVAE d={d_thresh:.1f}:\n')
            f.write(f'  Threshold: {d_thresh}\n')
            f.write(f'  Edges: {gs["undirected_edges"]:,}\n')
            f.write(f'  Connected: {gs["connected"]:,} ({gs["pct_connected"]:.1f}%)\n')
            for k in ['n_clusters', 'n_singletons', 'n_clustered',
                       'gc_median', 'gc_mean', 'gc_weighted_mean',
                       'gc_p75', 'gc_p90', 'pct_lt5']:
                f.write(f'  {k}: {cs[k]}\n')

    print(f'\nSummary saved to {summary_path}')


if __name__ == '__main__':
    main()
