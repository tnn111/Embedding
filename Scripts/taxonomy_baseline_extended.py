#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "scikit-learn",
# ]
# ///
"""
Extended taxonomic coherence comparison: VAE vs PCA vs CLR clustering.

Computes:
1. V-measure (harmonic mean of homogeneity and completeness)
2. Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI)
3. Pairwise F1 (precision = purity-like, recall = completeness-like)
4. Fragmentation distribution (clusters per species)
5. Cross-sample coherence (SFE vs SE contigs of same species)

Uses GTDB-Tk direct classifications only (not propagated).
"""

import os
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
)

MIN_LENGTH = 100_000
TAXONOMY_DIR = 'Runs/taxonomy'
RANKS = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']


def load_gtdbtk_classifications() -> dict[str, dict[str, str]]:
    """Load all GTDB-Tk classifications."""
    classified = {}
    files = [
        f'{TAXONOMY_DIR}/SFE_gtdbtk_output/classify/gtdbtk.bac120.summary.tsv',
        f'{TAXONOMY_DIR}/SFE_gtdbtk_output/classify/gtdbtk.ar53.summary.tsv',
        f'{TAXONOMY_DIR}/SE_gtdbtk_output/classify/gtdbtk.bac120.summary.tsv',
        f'{TAXONOMY_DIR}/SE_gtdbtk_output/classify/gtdbtk.ar53.summary.tsv',
    ]
    for filepath in files:
        if not os.path.exists(filepath):
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
                ranks = classification.split(';')
                taxonomy = {}
                for rank_str, rank_name in zip(ranks, RANKS):
                    value = rank_str[3:] if len(rank_str) > 3 else ''
                    if value:
                        taxonomy[rank_name] = value
                if 'domain' in taxonomy:
                    classified[contig_id] = taxonomy
    return classified


def load_clusters_as_map(cluster_file: str,
                         valid_ids: set[str]) -> dict[str, int]:
    """Load MCL clusters and return contig -> cluster_id mapping."""
    contig_to_cluster = {}
    cluster_id = 0
    with open(cluster_file) as f:
        for line in f:
            members = [m for m in line.strip().split('\t') if m in valid_ids]
            for m in members:
                contig_to_cluster[m] = cluster_id
            cluster_id += 1
    return contig_to_cluster


def compute_sklearn_metrics(contig_to_cluster: dict[str, int],
                            classified: dict[str, dict[str, str]],
                            rank: str) -> dict:
    """Compute V-measure, ARI, NMI for contigs classified at given rank."""
    # Need contigs that are both clustered and classified at this rank
    contigs = [c for c in contig_to_cluster
                if c in classified and rank in classified[c]]

    if len(contigs) < 2:
        return {}

    cluster_labels = [contig_to_cluster[c] for c in contigs]
    taxon_labels = [classified[c][rank] for c in contigs]

    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
        taxon_labels, cluster_labels)
    ari = adjusted_rand_score(taxon_labels, cluster_labels)
    nmi = normalized_mutual_info_score(taxon_labels, cluster_labels)

    return {
        'n_contigs': len(contigs),
        'homogeneity': homogeneity,
        'completeness': completeness,
        'v_measure': v_measure,
        'ari': ari,
        'nmi': nmi,
    }


def compute_pairwise_f1(contig_to_cluster: dict[str, int],
                        classified: dict[str, dict[str, str]],
                        rank: str,
                        max_pairs: int = 500_000) -> dict:
    """Compute pairwise precision, recall, F1.

    For efficiency, samples pairs if total would exceed max_pairs.
    A pair is:
    - True positive: same taxon AND same cluster
    - False positive: different taxon AND same cluster
    - False negative: same taxon AND different cluster
    """
    contigs = [c for c in contig_to_cluster
                if c in classified and rank in classified[c]]

    if len(contigs) < 2:
        return {}

    n_total_pairs = len(contigs) * (len(contigs) - 1) // 2

    cluster_labels = np.array([contig_to_cluster[c] for c in contigs])
    taxon_labels = np.array([classified[c][rank] for c in contigs])

    # For large sets, sample pairs
    if n_total_pairs > max_pairs:
        rng = np.random.default_rng(42)
        n = len(contigs)
        # Sample random pairs by generating random indices
        idx_a = rng.integers(0, n, size = max_pairs)
        idx_b = rng.integers(0, n - 1, size = max_pairs)
        # Avoid self-pairs
        idx_b[idx_b >= idx_a] += 1

        same_taxon = taxon_labels[idx_a] == taxon_labels[idx_b]
        same_cluster = cluster_labels[idx_a] == cluster_labels[idx_b]
    else:
        # Compute all pairs
        n = len(contigs)
        idx_a, idx_b = np.triu_indices(n, k = 1)
        same_taxon = taxon_labels[idx_a] == taxon_labels[idx_b]
        same_cluster = cluster_labels[idx_a] == cluster_labels[idx_b]

    tp = np.sum(same_taxon & same_cluster)
    fp = np.sum(~same_taxon & same_cluster)
    fn = np.sum(same_taxon & ~same_cluster)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'n_pairs': int(np.sum(same_taxon) + np.sum(~same_taxon)),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'sampled': n_total_pairs > max_pairs,
    }


def compute_fragmentation(contig_to_cluster: dict[str, int],
                          classified: dict[str, dict[str, str]],
                          rank: str,
                          min_members: int = 5) -> dict:
    """For each taxon with >= min_members classified contigs, count clusters."""
    # Group contigs by taxon
    taxon_contigs = defaultdict(list)
    for c in contig_to_cluster:
        if c in classified and rank in classified[c]:
            taxon_contigs[classified[c][rank]].append(c)

    n_clusters_per_taxon = []
    taxon_details = []

    for taxon, contigs in sorted(taxon_contigs.items(),
                                  key = lambda x: len(x[1]), reverse = True):
        if len(contigs) < min_members:
            continue

        clusters_used = set(contig_to_cluster[c] for c in contigs)
        n_clusters = len(clusters_used)
        n_clusters_per_taxon.append(n_clusters)

        # Largest cluster fraction
        cluster_counts = Counter(contig_to_cluster[c] for c in contigs)
        largest = cluster_counts.most_common(1)[0][1]
        largest_frac = largest / len(contigs)

        taxon_details.append({
            'taxon': taxon,
            'n_contigs': len(contigs),
            'n_clusters': n_clusters,
            'largest_frac': largest_frac,
        })

    arr = np.array(n_clusters_per_taxon)
    return {
        'n_taxa': len(arr),
        'median_clusters': np.median(arr) if len(arr) > 0 else 0,
        'mean_clusters': np.mean(arr) if len(arr) > 0 else 0,
        'pct_single': 100 * np.mean(arr == 1) if len(arr) > 0 else 0,
        'pct_le3': 100 * np.mean(arr <= 3) if len(arr) > 0 else 0,
        'distribution': arr,
        'details': taxon_details,
    }


def compute_cross_sample(contig_to_cluster: dict[str, int],
                         classified: dict[str, dict[str, str]],
                         rank: str,
                         min_per_sample: int = 2) -> dict:
    """For taxa present in both SFE and SE, check if they co-cluster."""
    # Group by taxon and sample
    taxon_sample_contigs = defaultdict(lambda: defaultdict(list))
    for c in contig_to_cluster:
        if c in classified and rank in classified[c]:
            sample = 'SFE' if c.startswith('SFE') else 'SE'
            taxon_sample_contigs[classified[c][rank]][sample].append(c)

    n_shared_taxa = 0
    n_co_clustered = 0
    details = []

    for taxon, samples in taxon_sample_contigs.items():
        if len(samples) < 2:
            continue
        if (len(samples.get('SFE', [])) < min_per_sample or
                len(samples.get('SE', [])) < min_per_sample):
            continue

        n_shared_taxa += 1

        # Check if any SFE and SE contigs share a cluster
        sfe_clusters = set(contig_to_cluster[c] for c in samples['SFE'])
        se_clusters = set(contig_to_cluster[c] for c in samples['SE'])
        overlap = sfe_clusters & se_clusters

        if overlap:
            n_co_clustered += 1

        # What fraction of contigs are in a shared cluster?
        all_contigs = samples['SFE'] + samples['SE']
        shared_cluster_contigs = [
            c for c in all_contigs if contig_to_cluster[c] in overlap
        ]
        frac_shared = len(shared_cluster_contigs) / len(all_contigs)

        details.append({
            'taxon': taxon,
            'n_sfe': len(samples['SFE']),
            'n_se': len(samples['SE']),
            'n_shared_clusters': len(overlap),
            'frac_in_shared': frac_shared,
        })

    details.sort(key = lambda x: x['n_sfe'] + x['n_se'], reverse = True)

    return {
        'n_shared_taxa': n_shared_taxa,
        'n_co_clustered': n_co_clustered,
        'pct_co_clustered': (100 * n_co_clustered / n_shared_taxa
                             if n_shared_taxa > 0 else 0),
        'details': details,
    }


def main():
    # ================================================================
    # Load data
    # ================================================================
    print('Loading GTDB-Tk classifications...')
    classified = load_gtdbtk_classifications()
    print(f'  {len(classified):,} classified contigs')

    print('Loading sequence data...')
    kmers_mmap = np.load('Runs/kmers_SFE_SE_1.npy', mmap_mode = 'r')
    long_mask = np.array(kmers_mmap[:, 0] >= MIN_LENGTH)
    long_indices = np.where(long_mask)[0]

    with open('Runs/ids_SFE_SE_1.txt') as f:
        all_ids = [l.strip() for l in f]
    valid_ids = set(all_ids[i] for i in long_indices)
    print(f'  {len(valid_ids):,} sequences >= {MIN_LENGTH:,} bp')

    classified_valid = {k: v for k, v in classified.items() if k in valid_ids}
    print(f'  {len(classified_valid):,} classified in >= {MIN_LENGTH:,} bp set')

    # ================================================================
    # Load clusterings
    # ================================================================
    cluster_files = {
        'VAE d=5.0': 'Runs/MCL_100_NCBI_5_d5/mcl_I3.0.clusters',
        'VAE d=4.9': 'Runs/PCA_baseline/VAE_d4.9/mcl_I3.0.clusters',
        'PCA-384': 'Runs/PCA_baseline/PCA_384/mcl_I3.0.clusters',
        'CLR-2772': 'Runs/PCA_baseline/CLR_2772/mcl_I3.0.clusters',
    }

    cluster_maps = {}
    for method, cfile in cluster_files.items():
        if not os.path.exists(cfile):
            print(f'  WARNING: {cfile} not found, skipping {method}')
            continue
        cluster_maps[method] = load_clusters_as_map(cfile, valid_ids)
        n_clustered = len(cluster_maps[method])
        print(f'  {method}: {n_clustered:,} clustered contigs')

    methods = list(cluster_maps.keys())

    # ================================================================
    # 1. V-measure, ARI, NMI
    # ================================================================
    print(f'\n{"=" * 70}')
    print('1. V-MEASURE, ARI, NMI')
    print('=' * 70)

    key_ranks = ['phylum', 'class', 'order', 'family', 'genus', 'species']

    for rank in key_ranks:
        print(f'\n  {rank.upper()}:')
        header = (f'    {"Method":>14s}  {"Homogeneity":>11s}  '
                  f'{"Completeness":>12s}  {"V-measure":>9s}  '
                  f'{"ARI":>7s}  {"NMI":>7s}')
        print(header)
        print(f'    {"-" * len(header.strip())}')

        for method in methods:
            r = compute_sklearn_metrics(
                cluster_maps[method], classified_valid, rank)
            if not r:
                continue
            print(f'    {method:>14s}  {r["homogeneity"]:>11.4f}  '
                  f'{r["completeness"]:>12.4f}  {r["v_measure"]:>9.4f}  '
                  f'{r["ari"]:>7.4f}  {r["nmi"]:>7.4f}')

    # ================================================================
    # 2. Pairwise F1
    # ================================================================
    print(f'\n{"=" * 70}')
    print('2. PAIRWISE F1')
    print('=' * 70)

    for rank in key_ranks:
        print(f'\n  {rank.upper()}:')
        header = (f'    {"Method":>14s}  {"Precision":>9s}  '
                  f'{"Recall":>7s}  {"F1":>7s}')
        print(header)
        print(f'    {"-" * len(header.strip())}')

        for method in methods:
            r = compute_pairwise_f1(
                cluster_maps[method], classified_valid, rank)
            if not r:
                continue
            sampled = ' (sampled)' if r['sampled'] else ''
            print(f'    {method:>14s}  {r["precision"]:>9.4f}  '
                  f'{r["recall"]:>7.4f}  {r["f1"]:>7.4f}{sampled}')

    # ================================================================
    # 3. Fragmentation distribution
    # ================================================================
    print(f'\n{"=" * 70}')
    print('3. FRAGMENTATION (clusters per taxon, taxa with >= 5 members)')
    print('=' * 70)

    for rank in ['genus', 'species']:
        print(f'\n  {rank.upper()}:')
        header = (f'    {"Method":>14s}  {"N taxa":>7s}  '
                  f'{"Median":>7s}  {"Mean":>7s}  '
                  f'{"Single":>7s}  {"<= 3":>7s}')
        print(header)
        print(f'    {"-" * len(header.strip())}')

        all_frag = {}
        for method in methods:
            r = compute_fragmentation(
                cluster_maps[method], classified_valid, rank)
            all_frag[method] = r
            print(f'    {method:>14s}  {r["n_taxa"]:>7d}  '
                  f'{r["median_clusters"]:>7.1f}  {r["mean_clusters"]:>7.1f}  '
                  f'{r["pct_single"]:>6.1f}%  {r["pct_le3"]:>6.1f}%')

        # Show most fragmented species for each method
        if rank == 'species':
            print(f'\n  Most fragmented species (top 10 by cluster count):')
            for method in methods:
                details = all_frag[method]['details']
                details_sorted = sorted(details,
                                         key = lambda x: x['n_clusters'],
                                         reverse = True)
                print(f'\n    {method}:')
                for d in details_sorted[:10]:
                    print(f'      {d["taxon"]:<40s}  '
                          f'{d["n_contigs"]:>4d} contigs  '
                          f'{d["n_clusters"]:>3d} clusters  '
                          f'largest={d["largest_frac"]:.1%}')

    # ================================================================
    # 4. Cross-sample coherence
    # ================================================================
    print(f'\n{"=" * 70}')
    print('4. CROSS-SAMPLE COHERENCE (taxa in both SFE and SE)')
    print('=' * 70)

    for rank in ['genus', 'species']:
        print(f'\n  {rank.upper()} (min 2 contigs per sample per taxon):')
        header = (f'    {"Method":>14s}  {"Shared taxa":>11s}  '
                  f'{"Co-clustered":>12s}  {"% co-clust":>10s}')
        print(header)
        print(f'    {"-" * len(header.strip())}')

        all_cross = {}
        for method in methods:
            r = compute_cross_sample(
                cluster_maps[method], classified_valid, rank)
            all_cross[method] = r
            print(f'    {method:>14s}  {r["n_shared_taxa"]:>11d}  '
                  f'{r["n_co_clustered"]:>12d}  '
                  f'{r["pct_co_clustered"]:>9.1f}%')

        # Top shared taxa details
        if rank == 'species':
            print(f'\n  Top 10 shared species (by total contigs):')
            # Use VAE as reference for listing
            ref = all_cross[methods[0]]['details'][:10]
            for d in ref:
                taxon = d['taxon']
                print(f'    {taxon} ({d["n_sfe"]} SFE + {d["n_se"]} SE):')
                for method in methods:
                    md = next((x for x in all_cross[method]['details']
                               if x['taxon'] == taxon), None)
                    if md:
                        print(f'      {method:>14s}: '
                              f'{md["n_shared_clusters"]} shared clusters, '
                              f'{md["frac_in_shared"]:.1%} contigs in shared')

    # ================================================================
    # Summary
    # ================================================================
    print(f'\n\n{"=" * 70}')
    print('SUMMARY TABLE')
    print('=' * 70)

    print(f'\n{"Metric":>30s}', end = '')
    for method in methods:
        print(f'  {method:>12s}', end = '')
    print()
    print('-' * (32 + 14 * len(methods)))

    # V-measure at species
    for rank in ['genus', 'species']:
        row_label = f'V-measure ({rank})'
        print(f'{row_label:>30s}', end = '')
        for method in methods:
            r = compute_sklearn_metrics(
                cluster_maps[method], classified_valid, rank)
            print(f'  {r["v_measure"]:>12.4f}', end = '')
        print()

    # ARI at species
    for rank in ['genus', 'species']:
        row_label = f'ARI ({rank})'
        print(f'{row_label:>30s}', end = '')
        for method in methods:
            r = compute_sklearn_metrics(
                cluster_maps[method], classified_valid, rank)
            print(f'  {r["ari"]:>12.4f}', end = '')
        print()

    # Pairwise F1 at species
    for rank in ['genus', 'species']:
        row_label = f'Pairwise F1 ({rank})'
        print(f'{row_label:>30s}', end = '')
        for method in methods:
            r = compute_pairwise_f1(
                cluster_maps[method], classified_valid, rank)
            print(f'  {r["f1"]:>12.4f}', end = '')
        print()

    # Cross-sample at species
    row_label = 'Cross-sample co-clust (species)'
    print(f'{row_label:>30s}', end = '')
    # Recompute is cheap for this
    for method in methods:
        r = compute_cross_sample(
            cluster_maps[method], classified_valid, 'species')
        print(f'  {r["pct_co_clustered"]:>11.1f}%', end = '')
    print()

    # Fragmentation at species
    row_label = 'Single-cluster species (%)'
    print(f'{row_label:>30s}', end = '')
    for method in methods:
        r = compute_fragmentation(
            cluster_maps[method], classified_valid, 'species')
        print(f'  {r["pct_single"]:>11.1f}%', end = '')
    print()


if __name__ == '__main__':
    main()
