# calculate_kmer_frequencies Development Log

## Original Specification

- Take as input one or more FASTA format files specified on the command line via an -i/--input argument which can take multiple file names.
- Produce as output two files:
  - `-id/--identifier`: Text file with one ID per line
  - `-k/--kmer`: NumPy array (.npy) containing length + canonical k-mers (7 through 1)

---

## 2025-11-30 ~16:45: Rewrote script to match specification

### Changes from previous version

The previous version output to stdout in text format with 6-mers through 3-mers + GC. Rewrote to:

1. **Input**: Multiple FASTA files via `-i/--input` (removed stdin support)
2. **Output**: Two files instead of stdout
   - `-id/--identifier`: Text file, one sequence ID per line
   - `-k/--kmer`: NumPy `.npy` file with float32 array
3. **K-mer range**: 7-mers through 1-mers (was 6-mers through 3-mers)
4. **Removed GC**: Canonical 1-mers (A/T and C/G) encode GC content as two numbers

### Output format

NumPy array shape: `(n_sequences, 10965)`

| Column(s) | Content | Count |
|-----------|---------|-------|
| 0 | Sequence length | 1 |
| 1-8192 | 7-mer frequencies | 8,192 |
| 8193-10272 | 6-mer frequencies | 2,080 |
| 10273-10784 | 5-mer frequencies | 512 |
| 10785-10920 | 4-mer frequencies | 136 |
| 10921-10952 | 3-mer frequencies | 32 |
| 10953-10962 | 2-mer frequencies | 10 |
| 10963-10964 | 1-mer frequencies | 2 |

### Usage

```bash
./calculate_kmer_frequencies -i file1.fasta file2.fasta -id ids.txt -k kmers.npy
```

### Notes

- Each k-mer group is normalized separately (sums to 1.0)
- K-mers are canonical (lexicographically smallest of k-mer and reverse complement)
- Sequences are filtered to ATGC only before processing
- Progress reported every 100,000 sequences

### Potential improvements to consider

1. **Parallelization**: K-mer counting is CPU-bound and embarrassingly parallel. Could use multiprocessing, but would require a producer-consumer queue model to avoid loading all sequences into memory. Not worth the complexity for infrequent runs.

2. **Performance**: The pure Python k-mer counting is relatively slow. Options:
   - **Numba**: Encode DNA as integers (A=0, C=1, G=2, T=3), precompute canonical mapping as NumPy array, JIT-compile the sliding window. Expected 50-100x speedup.
   - **Jellyfish/KMC**: Fast C++ k-mer counters, but they count globally across files, not per-sequence. Would need to split each sequence to a temp file—overhead likely negates benefit for many short sequences.

---

## 2025-11-30 ~17:15: Fixed memory issue with chunked processing

### Problem

Original implementation accumulated all features in Python lists before converting to NumPy. Python floats are 24-byte objects, so each sequence used ~250KB instead of ~44KB as float32. For 5M sequences: ~1.2TB in Python vs ~200GB in NumPy.

### Solution

Chunked approach:
1. Process 100,000 sequences at a time
2. Convert chunk to NumPy float32 array
3. Write to temp file (`{output}.tmp.N`)
4. Clear Python list, repeat
5. At end, load all temp files and concatenate
6. Delete temp files

### Memory usage

- During processing: ~5-10GB (one chunk in Python lists + one as NumPy)
- During final concatenation: ~200GB for 5M sequences (loads all chunks)
- 200GB fits comfortably in 512GB RAM

### Other changes

- IDs written immediately to file (no accumulation)
- Progress reports every chunk (100,000 sequences)

---

## 2025-11-30 ~18:00: Memory leak fix - forced garbage collection

### Problem

Even with pre-allocated NumPy arrays and chunked processing, memory grew unbounded (~500KB per sequence). Reached 256GB+ before processing 100k sequences.

### Investigation

Added memory tracking every 1000 sequences with `gc.collect()`. Memory stabilized at ~44GB (8.5% of 512GB).

### Root cause

Python's automatic garbage collector wasn't keeping up with the rate of temporary object creation. Each sequence creates:
- Encoded sequence array
- 7 count arrays (one per k-mer size)
- 7 frequency arrays
- Various intermediate strings from FASTA parsing

These are short-lived but created rapidly. Python's generational GC wasn't triggering often enough.

### Solution

Force `gc.collect()` every 10,000 sequences. Memory now stays flat at ~44GB.

### Other optimizations applied during debugging

1. **Integer-encoded k-mer counting**: Replaced string slicing with integer arithmetic
   - Sequences encoded as NumPy uint8 arrays (A=0, C=1, G=2, T=3)
   - K-mers represented as integers via bit shifting
   - Canonical mapping via pre-computed NumPy array lookup (not Python dict)

2. **Simplified FASTA reader**: Removed `groupby()` which may have held references

3. **Explicit deletion**: `del` temporary arrays after use (may help GC identify garbage sooner)

---

## 2025-11-30 ~18:30: Clean rewrite

### Context

After extensive debugging of memory issues, rewrote the script from scratch with a cleaner architecture. The previous iterations had accumulated complexity from debugging attempts.

### Final architecture

1. **Pre-built canonical mappings**: For k=1 through 7, build NumPy arrays that map k-mer integers directly to canonical indices. Done once at startup.

2. **Integer-encoded sequences**: DNA sequences converted to uint8 arrays (A=0, C=1, G=2, T=3) immediately after reading.

3. **Efficient k-mer counting**: Sliding window using bit operations:
   - Initial k-mer built by shifting: `kmer_int = (kmer_int << 2) | base`
   - Subsequent k-mers: `kmer_int = ((kmer_int << 2) | next_base) & mask`
   - Direct array lookup for canonical index: `counts[canonical_map[kmer_int]] += 1`

4. **Chunked output**:
   - Pre-allocate float32 array for 100,000 sequences
   - Write to temp files when chunk is full
   - Concatenate all chunks at end
   - `gc.collect()` after each chunk write

### Output verification

Total features: 10,965 (was incorrectly stated as 10,963 in earlier docs)
- 1 length
- 8,192 canonical 7-mers
- 2,080 canonical 6-mers
- 512 canonical 5-mers
- 136 canonical 4-mers
- 32 canonical 3-mers
- 10 canonical 2-mers
- 2 canonical 1-mers (A/T combined, C/G combined)

Note: 2-mers have 10 canonical forms, not 8 as mistakenly calculated earlier.

---

## 2025-11-30 ~19:00: Vectorized k-mer counting

### Problem

Script showed zero progress and no memory usage. Root cause: pure Python loop in `count_kmers` was too slow for ~5000 bp sequences. Each sequence required ~35,000 Python loop iterations (5000 positions × 7 k-mer sizes).

### Solution

Replaced Python loop with vectorized NumPy operations:

```python
# Build all k-mer integers at once using array slicing
n_kmers = len(seq_encoded) - k + 1
kmer_ints = np.zeros(n_kmers, dtype=np.int32)
for i in range(k):
    kmer_ints += seq_encoded[i:i + n_kmers].astype(np.int32) << (2 * (k - 1 - i))

# Count using np.bincount
canonical_indices = canonical_map[kmer_ints]
counts = np.bincount(canonical_indices, minlength=n_canonical)
```

### Other changes

- Restored `groupby`-based FASTA reader (cleaner than manual state machine)
- Chunks kept in memory instead of temp files (512GB RAM is plenty)
- Removed `gc.collect()` calls (no longer needed)

### Performance

Processed 4,776,770 sequences in ~1 hour. Output: (4776770, 10965) float32 array.