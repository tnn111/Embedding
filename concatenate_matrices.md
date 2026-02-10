# concatenate_matrices Development Log

## 2026-02-04: Initial implementation

### Purpose

Concatenates multiple k-mer frequency matrices (.npy) and their corresponding ID files (.txt) into single output files. Designed to work with output from `calculate_kmer_frequencies`.

### Usage

```bash
./concatenate_matrices \
    -i file1.npy file2.npy file3.npy \
    -id file1.txt file2.txt file3.txt \
    -o combined
```

This produces:
- `combined.npy` - concatenated matrix
- `combined.txt` - concatenated IDs

### Features

- **Memory-efficient**: Uses memmap to create output array without loading all data into memory
- **Validation**: Checks that each .npy file has same number of rows as lines in corresponding .txt file
- **Gzip support**: ID files can be gzip-compressed (.txt.gz)
- **Progress reporting**: Shows progress during validation and concatenation

### Implementation details

- Reads .npy headers without loading data to get shapes
- Creates output memmap with total row count
- Copies one file at a time to output memmap
- Frees memory after each file with `del data`
- Assumes all .npy files have same number of columns (features) as first file

### Memory usage

Peak memory is approximately the size of the largest input .npy file, not the total output size.

## 2026-02-09: Added --shuffle flag

### Problem

The VAE uses a contiguous 90/10 train/val split (first 90% train, last 10% val) and assumes pre-shuffled data. When individual source datasets are shuffled internally but then concatenated in order, the validation set ends up dominated by whichever source was last — creating a systematic distribution shift between train and val.

### Solution

Added `--shuffle` flag that, after concatenation, generates a random permutation and applies it to both the matrix rows and ID lines, ensuring uniform mixing of all sources.

### Implementation

- Loads the entire memmap into RAM, applies the permutation, writes back
- Shuffles IDs with the same permutation to maintain row correspondence
- Peak memory with `--shuffle`: ~2x the output matrix size (memmap + RAM copy)
