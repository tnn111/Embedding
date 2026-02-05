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
