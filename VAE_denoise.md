# VAE_denoise.py Notes

## Overview
Denoising variant of VAE.py. Same architecture (2772→384 latent). Two noise
modes: (1) Dirichlet on-the-fly noise simulating shorter contigs, or
(2) pre-computed nucleotide-level corruption pairs via `corrupt_and_count`.

## Key Difference from VAE.py
- Training input is concatenated [encoder_input, clean_target] (5544-dim)
- Model's `call()` slices to route encoder input and compute loss vs clean
- Encoder/decoder retain standard 2772/384-dim interfaces for inference

## Changes

### 2026-03-14: Initial implementation
- Standalone denoising VAE with Dirichlet noise calibrated to simulated
  shorter contigs (min 5 kbp, log-uniform sampling)
- DenoisingBatchDataset generates noisy→clean pairs
- CleanBatchDataset for validation (clean→clean)
- Tested end-to-end with 20K samples / 3 epochs

### 2026-03-15: Dual clean+noisy training
- Modified DenoisingBatchDataset to interleave noisy and clean batches
- Even batch indices (0, 2, 4, ...) → noisy→clean
- Odd batch indices (1, 3, 5, ...) → clean→clean (same data slice)
- `__len__` returns `2 * n_data_batches` — each sample appears twice per epoch
- Renamed internal `n_batches` to `n_data_batches` for clarity
- Training time doubles per epoch (~2 min/epoch on NCBI_5)
- Rationale: at inference, long contigs have clean profiles but the
  noisy-only model never trained on clean input. Clean is just the
  zero-noise end of the continuum.

### 2026-03-15: Pre-computed corruption mode
- Added `--corrupted` flag for pre-computed paired data from `corrupt_and_count`
- New `CorruptedPairBatchDataset`: loads clean + corrupted CLR matrices,
  interleaves [corrupted→clean] and [clean→clean] batches (same pattern
  as Dirichlet dual mode)
- Shuffles both matrices with same permutation at load time (data may not
  be pre-shuffled since contigs come from multiple FASTA files in order)
- No on-the-fly noise generation — simpler, faster, more biologically
  realistic (real k-mer artifacts from nucleotide substitutions)
- Dirichlet mode remains the default when `--corrupted` not provided

## Related Scripts
- **corrupt_and_count**: Generates paired clean/corrupted k-mer matrices
  from FASTA. Introduces random substitution errors (log-uniform 0.1-2%),
  computes canonical k-mer frequencies for both versions. PEP 723 standalone.

## Run History
- **Run_NCBI_5_denoise**: noisy-only (Dirichlet), 1000 epochs, final
  val_loss 83.28, Spearman 0.840 on SFE_SE_5 (baseline NCBI_5: 0.837)
- **Run_NCBI_5_denoise_dual**: dual clean+noisy (Dirichlet), training
- **Run_NCBI_euk_corrupt**: nucleotide-level corruption, NCBI prokaryotic
  + eukaryotic genomes (~668K sequences). Data generation pending.
