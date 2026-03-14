#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
# ]
# ///
"""
Compare taxonomic coherence of VAE vs PCA clustering.

Uses GTDB-Tk direct classifications (not propagated) to evaluate
cluster purity and taxon completeness for:
- VAE-384 at d=5.0 (original, 3.55M edges)
- VAE-384 at d=4.9 (edge-matched with PCA, 2.19M edges)
- PCA-384 at d=4.62 (connectivity-matched, 2.21M edges)

Purity: for each cluster with >= 2 classified members, what fraction
share the dominant taxon at each rank?

Completeness: for each taxon with >= 2 members, what fraction of its
members are in the dominant cluster?
"""

import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

MIN_LENGTH = 100_000
TAXONOMY_DIR = 'Runs/taxonomy'
RANKS = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']
RANK_PREFIXES = ['d__', 'p__', 'c__', 'o__', 'f__', 'g__', 's__']


def load_gtdbtk_classifications() -> dict[str, dict[str, str]]:
    """Load all GTDB-Tk classifications from SFE and SE output directories.

    Returns dict mapping contig_id -> {rank: taxon_name} for classified contigs.
    Only includes contigs with at least domain-level classification.
    """
    classified = {}

    files = [
        f'{TAXONOMY_DIR}/SFE_gtdbtk_output/classify/gtdbtk.bac120.summary.tsv',
        f'{TAXONOMY_DIR}/SFE_gtdbtk_output/classify/gtdbtk.ar53.summary.tsv',
        f'{TAXONOMY_DIR}/SE_gtdbtk_output/classify/gtdbtk.bac120.summary.tsv',
        f'{TAXONOMY_DIR}/SE_gtdbtk_output/classify/gtdbtk.ar53.summary.tsv',
    ]

    for filepath in files:
        if not os.path.exists(filepath):
            print(f'  WARNING: {filepath} not found, skipping')
            continue

        with open(filepath) as f:
            header = f.readline().strip().split('\t')
            genome_col = header.index('user_genome')
            class_col = header.index('classification')

            for line in f:
                fields = line.strip().split('\t')
                contig_id = fields[genome_col]
                classification = fields[class_col]

                if classification.startswith('Unclassified') or not classification:
                    continue

                # Parse "d__Bacteria;p__Phylum;c__Class;..."
                ranks = classification.split(';')
                taxonomy = {}
                for rank_str, rank_name in zip(ranks, RANKS):
                    # Remove prefix (d__, p__, etc.)
                    value = rank_str[3:] if len(rank_str) > 3 else ''
                    if value:
                        taxonomy[rank_name] = value

                if 'domain' in taxonomy:
                    classified[contig_id] = taxonomy

    return classified


def load_clusters(cluster_file: str, valid_ids: set[str]) -> list[list[str]]:
    """Load MCL clusters, filtering to valid IDs. Returns list of member lists."""
    clusters = []
    with open(cluster_file) as f:
        for line in f:
            members = [m for m in line.strip().split('\t') if m in valid_ids]
            if len(members) >= 2:
                clusters.append(members)
    clusters.sort(key = len, reverse = True)
    return clusters


def compute_purity(clusters: list[list[str]],
                   classified: dict[str, dict[str, str]]) -> dict:
    """Compute cluster purity at each taxonomic rank.

    For each cluster with >= 2 classified members, purity at rank R =
    count(dominant_taxon) / count(classified_members_with_rank_R).
    """
    results = {}

    for rank in RANKS:
        purities = []
        n_perfect = 0
        n_evaluated = 0

        for members in clusters:
            # Get classified members at this rank
            taxa = []
            for m in members:
                if m in classified and rank in classified[m]:
                    taxa.append(classified[m][rank])

            if len(taxa) < 2:
                continue

            n_evaluated += 1
            counts = Counter(taxa)
            dominant_count = counts.most_common(1)[0][1]
            purity = dominant_count / len(taxa)
            purities.append(purity)
            if purity == 1.0:
                n_perfect += 1

        purities = np.array(purities)
        results[rank] = {
            'mean': np.mean(purities) if len(purities) > 0 else 0,
            'median': np.median(purities) if len(purities) > 0 else 0,
            'n_perfect': n_perfect,
            'n_evaluated': n_evaluated,
            'pct_perfect': 100 * n_perfect / n_evaluated if n_evaluated > 0 else 0,
        }

    return results


def compute_completeness(clusters: list[list[str]],
                         classified: dict[str, dict[str, str]]) -> dict:
    """Compute taxon completeness at each rank.

    For each taxon with >= 2 classified members in the clustering,
    completeness = count_in_largest_cluster / total_count.
    """
    results = {}

    for rank in RANKS:
        # Map taxon -> list of cluster indices
        taxon_clusters = defaultdict(list)
        for ci, members in enumerate(clusters):
            for m in members:
                if m in classified and rank in classified[m]:
                    taxon_clusters[classified[m][rank]].append(ci)

        completenesses = []
        n_perfect = 0
        n_evaluated = 0

        for taxon, cluster_indices in taxon_clusters.items():
            if len(cluster_indices) < 2:
                continue

            n_evaluated += 1
            counts = Counter(cluster_indices)
            largest = counts.most_common(1)[0][1]
            completeness = largest / len(cluster_indices)
            completenesses.append(completeness)
            if completeness == 1.0:
                n_perfect += 1

        completenesses = np.array(completenesses)
        results[rank] = {
            'mean': np.mean(completenesses) if len(completenesses) > 0 else 0,
            'median': np.median(completenesses) if len(completenesses) > 0 else 0,
            'n_perfect': n_perfect,
            'n_evaluated': n_evaluated,
            'pct_perfect': (100 * n_perfect / n_evaluated
                           if n_evaluated > 0 else 0),
        }

    return results


def print_comparison(metric_name: str, all_results: dict[str, dict],
                     ranks: list[str] = RANKS) -> None:
    """Print side-by-side comparison table."""
    methods = list(all_results.keys())

    print(f'\n{metric_name}')
    print('=' * (14 + 36 * len(methods)))

    # Header
    header = f'{"Rank":>12s}'
    for method in methods:
        header += f'  {method:>12s} perfect'
    print(header)
    print('-' * len(header))

    for rank in ranks:
        row = f'{rank:>12s}'
        for method in methods:
            r = all_results[method][rank]
            row += (f'  {r["mean"]:>6.1%} '
                    f'{r["n_perfect"]:>5d}/{r["n_evaluated"]:<5d} '
                    f'({r["pct_perfect"]:>5.1f}%)')
        print(row)


def main():
    # ================================================================
    # Load GTDB-Tk classifications
    # ================================================================
    print('Loading GTDB-Tk classifications...')
    classified = load_gtdbtk_classifications()
    print(f'  {len(classified):,} classified contigs')

    # Count by depth
    for rank in RANKS:
        n = sum(1 for t in classified.values() if rank in t)
        print(f'  {rank:>10s}: {n:,}')

    # ================================================================
    # Load IDs (100 kbp filter)
    # ================================================================
    print('\nLoading sequence data...')
    kmers_mmap = np.load('Runs/kmers_SFE_SE_1.npy', mmap_mode = 'r')
    long_mask = np.array(kmers_mmap[:, 0] >= MIN_LENGTH)
    long_indices = np.where(long_mask)[0]

    with open('Runs/ids_SFE_SE_1.txt') as f:
        all_ids = [l.strip() for l in f]
    valid_ids = set(all_ids[i] for i in long_indices)
    print(f'  {len(valid_ids):,} sequences >= {MIN_LENGTH:,} bp')

    # Filter classified to valid IDs
    classified_valid = {k: v for k, v in classified.items() if k in valid_ids}
    print(f'  {len(classified_valid):,} classified in >= {MIN_LENGTH:,} bp set')

    # ================================================================
    # Define clusterings to compare
    # ================================================================
    cluster_files = {
        'VAE d=5.0': 'Runs/MCL_100_NCBI_5_d5/mcl_I3.0.clusters',
        'VAE d=4.9': 'Runs/PCA_baseline/VAE_d4.9/mcl_I3.0.clusters',
        'PCA-384': 'Runs/PCA_baseline/PCA_384/mcl_I3.0.clusters',
        'CLR-2772': 'Runs/PCA_baseline/CLR_2772/mcl_I3.0.clusters',
    }

    all_purity = {}
    all_completeness = {}

    for method, cfile in cluster_files.items():
        if not os.path.exists(cfile):
            print(f'\n  WARNING: {cfile} not found, skipping {method}')
            continue

        print(f'\n{"=" * 70}')
        print(f'Evaluating: {method}')
        print('=' * 70)

        clusters = load_clusters(cfile, valid_ids)
        n_clustered = sum(len(c) for c in clusters)
        n_classified_in_clusters = sum(
            1 for c in clusters for m in c if m in classified_valid)

        print(f'  {len(clusters):,} non-singleton clusters, '
              f'{n_clustered:,} sequences')
        print(f'  {n_classified_in_clusters:,} classified members in clusters')

        purity = compute_purity(clusters, classified_valid)
        completeness = compute_completeness(clusters, classified_valid)

        all_purity[method] = purity
        all_completeness[method] = completeness

        # Print individual results
        print(f'\n  Purity (mean / perfect):')
        for rank in RANKS:
            r = purity[rank]
            print(f'    {rank:>10s}: {r["mean"]:.1%}  '
                  f'{r["n_perfect"]}/{r["n_evaluated"]} '
                  f'({r["pct_perfect"]:.1f}%)')

        print(f'\n  Completeness (mean / perfect):')
        for rank in RANKS:
            r = completeness[rank]
            print(f'    {rank:>10s}: {r["mean"]:.1%}  '
                  f'{r["n_perfect"]}/{r["n_evaluated"]} '
                  f'({r["pct_perfect"]:.1f}%)')

    # ================================================================
    # Side-by-side comparison
    # ================================================================
    print(f'\n\n{"=" * 70}')
    print('SIDE-BY-SIDE COMPARISON')
    print('=' * 70)

    print_comparison('PURITY (mean / n_perfect / n_evaluated / % perfect)',
                     all_purity)
    print_comparison('COMPLETENESS (mean / n_perfect / n_evaluated / % perfect)',
                     all_completeness)

    # ================================================================
    # Summary: key metrics
    # ================================================================
    print(f'\n\n{"=" * 70}')
    print('KEY METRICS SUMMARY')
    print('=' * 70)

    key_ranks = ['phylum', 'genus', 'species']
    for metric_name, all_results in [('Purity', all_purity),
                                      ('Completeness', all_completeness)]:
        print(f'\n{metric_name}:')
        header = f'{"Rank":>10s}'
        for method in all_results:
            header += f'  {method:>14s}'
        print(header)
        print('-' * len(header))
        for rank in key_ranks:
            row = f'{rank:>10s}'
            for method in all_results:
                r = all_results[method][rank]
                row += f'  {r["mean"]:>6.1%} ({r["pct_perfect"]:>5.1f}%)'
            print(row)

    # Save summary
    summary_path = 'Runs/PCA_baseline/taxonomy_comparison.txt'
    with open(summary_path, 'w') as f:
        f.write('Taxonomic Coherence: VAE vs PCA Clustering\n')
        f.write(f'Classified contigs in >= {MIN_LENGTH:,} bp set: '
                f'{len(classified_valid):,}\n\n')

        for metric_name, all_results in [('PURITY', all_purity),
                                          ('COMPLETENESS', all_completeness)]:
            f.write(f'{metric_name}\n')
            f.write(f'{"Rank":>12s}')
            for method in all_results:
                f.write(f'  {method:>30s}')
            f.write('\n')
            for rank in RANKS:
                f.write(f'{rank:>12s}')
                for method in all_results:
                    r = all_results[method][rank]
                    f.write(f'  {r["mean"]:.4f} '
                            f'({r["n_perfect"]}/{r["n_evaluated"]}, '
                            f'{r["pct_perfect"]:.1f}%)')
                f.write('\n')
            f.write('\n')

    print(f'\nSaved to {summary_path}')


if __name__ == '__main__':
    main()
