#!/usr/bin/env python3
"""
Calculate canonical 7-mer frequencies from metagenomic contigs.

Reads a FASTA file and outputs normalized 7-mer frequencies where:
- K-mers are canonical (lexicographically smallest of k-mer and reverse complement)
- Frequencies are normalized to sum to 1.0
- K-mers containing non-ATGC bases are skipped
- Output has 8,192 frequency values per sequence (4^7 / 2)
"""

import argparse
import sys
from collections import Counter
from itertools import groupby, product
from pathlib import Path
from typing import Iterator


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    return ''.join(complement[base] for base in reversed(seq))


def get_canonical_kmer(kmer: str) -> str:
    """Return the canonical form (lexicographically smallest) of a k-mer."""
    rc = reverse_complement(kmer)
    return min(kmer, rc)


def generate_canonical_kmers(k: int) -> list[str]:
    """
    Generate all canonical k-mers in lexicographic order.

    For k=7, this generates 8,192 canonical k-mers (4^7 / 2).
    """
    bases = 'ACGT'

    # Generate all possible k-mers using itertools.product
    all_kmers = [''.join(kmer) for kmer in product(bases, repeat = k)]

    # Get canonical forms and keep unique ones
    canonical_set = {get_canonical_kmer(kmer) for kmer in all_kmers}

    # Sort lexicographically
    return sorted(canonical_set)


def count_canonical_kmers(sequence: str, k: int) -> Counter:
    """
    Count canonical k-mers in a sequence.

    Assumes sequence contains only ATGC bases (filtered upstream).
    """
    counts = Counter()

    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        canonical = get_canonical_kmer(kmer)
        counts[canonical] += 1

    return counts


def read_fasta_sequences(fasta_file: Path) -> Iterator[str]:
    """
    Generator that yields sequences from a FASTA file one at a time.

    Memory-efficient for large files. Handles multi-line sequences.
    Uses itertools.groupby for elegant parsing.
    """
    try:
        with open(fasta_file) as f:
            # Group lines by whether they start with '>' (header) or not (sequence)
            groups = groupby(f, key = lambda line: line.startswith('>'))

            for is_header, group in groups:
                if is_header:
                    # Skip header line(s)
                    next(group, None)
                else:
                    # Join sequence lines, stripping whitespace
                    sequence = ''.join(line.strip() for line in group)
                    # Filter to keep only ATGC bases (more efficient than checking each k-mer)
                    sequence = ''.join(base for base in sequence.upper() if base in 'ATGC')
                    if sequence:  # Only yield non-empty sequences
                        yield sequence

    except FileNotFoundError:
        print(f'Error: File not found: {fasta_file}', file=sys.stderr)
        sys.exit(1)
    except IOError as e:
        print(f'Error reading file {fasta_file}: {e}', file=sys.stderr)
        sys.exit(1)


def calculate_normalized_frequencies(counts: Counter, canonical_kmers: list[str]) -> list[float]:
    """
    Calculate normalized frequencies that sum to 1.0.

    Returns frequencies in the same order as canonical_kmers.
    """
    total = sum(counts.values())

    if total == 0:
        # No valid k-mers found, return zeros
        return [0.0] * len(canonical_kmers)

    # Use list comprehension for efficiency
    return [counts.get(kmer, 0) / total for kmer in canonical_kmers]


def main():
    """Main function to process contigs and output k-mer frequencies."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description = 'Calculate canonical 7-mer frequencies from FASTA file'
    )
    parser.add_argument(
        '-i', '--input',
        required = True,
        type = Path,
        help = 'Input FASTA file'
    )
    args = parser.parse_args()

    # Configuration
    k = 7
    input_file = args.input

    # Validate input file
    if not input_file.exists():
        print(f'Error: Input file does not exist: {input_file}', file=sys.stderr)
        sys.exit(1)

    # Generate canonical k-mers in lexicographic order
    print(f'Generating canonical {k}-mers...', file=sys.stderr)
    canonical_kmers = generate_canonical_kmers(k)
    print(f'Generated {len(canonical_kmers)} canonical {k}-mers', file=sys.stderr)

    # Process sequences one at a time (memory efficient)
    print(f'Reading {input_file} and calculating k-mer frequencies...', file=sys.stderr)
    seq_count = 0

    for sequence in read_fasta_sequences(input_file):
        # Skip empty sequences
        if not sequence:
            continue

        # Count canonical k-mers
        counts = count_canonical_kmers(sequence, k)

        # Normalize to frequencies
        frequencies = calculate_normalized_frequencies(counts, canonical_kmers)

        # Format output: freq1 freq2 freq3 ... (no ID)
        freq_str = ' '.join(f'{freq:.10f}' for freq in frequencies)
        print(freq_str)

        seq_count += 1

    print(f'Done! Processed {seq_count} sequences', file=sys.stderr)


if __name__ == '__main__':
    main()
