# VAE_5mer.py Notes

## Overview
VAE using only 5-mer through 1-mer features (692-dim input, 128-dim latent).
Drops 6-mers entirely. 5-mers are more robust to assembly errors and have
better statistical coverage at short contig lengths. Outperforms the 6-mer
model on taxonomic coherence at every level.

## Architecture
- Encoder: 692 -> Dense(512) -> BN -> LeakyReLU -> Dense(256) -> BN -> LeakyReLU -> z_mean(128) / z_log_var(128)
- Decoder: 128 -> Dense(256) -> BN -> LeakyReLU -> Dense(512) -> BN -> LeakyReLU -> Dense(692)
- ~1.1M parameters (vs ~7.1M for 6-mer model)

## Rationale
- 5-mers: ~10 counts/bin at 5 kbp vs ~2.4 for 6-mers (4x better sampled)
- Each error corrupts 5 overlapping 5-mers vs 6 overlapping 6-mers
- 692:128 compression ratio (~5:1) similar to 2772:384 (~7:1) in original
- Uses same .npy data files as 6-mer model, just different column slice
  (columns 2081-2772 instead of 1-2772)

## Changes

### 2026-03-16: Initial implementation
- Based on VAE.py with reduced dimensions
- COL_START=2081, COL_END=2773 (skip length column and 6-mers)
- KMER_SIZES adjusted for 0-based local indices within 692-dim input
- verify_local_distances.py updated to auto-detect encoder input dim

### 2026-03-16: Warmup checkpoint bug fix
- VAECheckpoint now has `skip_epochs` parameter; skips saving during KL
  warmup to avoid artificially low val_loss from near-zero KL weight
- Bug surfaced on SFE_SE_5 data where reconstruction loss alone (~11)
  was lower than post-warmup val_loss (~15.9 with KL contribution ~4.7)
- Same bug exists in VAE.py and VAE_denoise.py (not yet fixed there)

## Results

### Per-dimension KL (NCBI_5mer, 128 dims)
- Total KL: 39.2
- Active dims (KL > 0.1): 35 / 128
- Near-zero (KL < 0.001): 56 / 128
- Top 10 carry 53.1% of total KL — highly concentrated

### Spearman on SFE_SE_5 (50K pool)
| Model | Training data | Spearman | CI |
|---|---|---|---|
| NCBI_5 6-mer (384d) | NCBI refs | 0.837 | — |
| SFE_SE_5 5-mer (128d) | Marine | 0.828 | 0.768-0.869 |
| NCBI_5 5-mer (128d) | NCBI refs | 0.772 | 0.692-0.830 |
| SFE_SE_5 6-mer (384d) | Marine | 0.692 | — |

### Taxonomic coherence (10-NN on NCBI reference genomes, 20K sample)
| Level | NCBI_5 6-mer | NCBI_5 5-mer | SFE_SE_5 5-mer |
|---|---|---|---|
| Domain | 0.993 | 0.995 | **0.996** |
| Phylum | 0.879 | 0.908 | **0.955** |
| Class | 0.833 | 0.865 | **0.913** |
| Order | 0.732 | 0.772 | **0.815** |
| Family | 0.647 | 0.683 | **0.706** |
| Genus | 0.465 | **0.514** | 0.504 |
| Species | 0.064 | **0.102** | 0.077 |

Both 5-mer models beat the 6-mer baseline at every level except species
(where only the NCBI 5-mer wins). Marine-trained dominates through family;
NCBI-trained wins at genus/species (has species-level ground truth).

### Taxonomic coherence at >= 100 kbp (10-NN on NCBI refs, 20K sample)
At long contig lengths, 6-mer and 5-mer models converge:

| Level | NCBI_5 6-mer | SFE_SE_5 5-mer | NCBI_5 5-mer |
|---|---|---|---|
| Phylum | 0.964 | **0.980** | 0.960 |
| Class | 0.945 | **0.960** | 0.936 |
| Order | 0.907 | **0.919** | 0.891 |
| Family | 0.845 | **0.848** | 0.824 |
| Genus | **0.655** | 0.644 | 0.647 |
| Species | 0.127 | 0.126 | **0.128** |

5-mer advantage is mostly at shorter lengths where 6-mers are undersampled.
At >= 100 kbp (~50 counts per 6-mer bin), the 6-mer model catches up at
genus/species. For deeply sequenced long-read metagenomes with mostly long
contigs, the 6-mer model may still be the better choice. The 5-mer model is
better for general-purpose use across assembly quality ranges.

### Full augmented 5-mer results
Mixing data sources hurts 5-mer models just as it did 6-mer models:
- Spearman: 0.709 (vs SFE_SE_5 5-mer 0.828)
- Taxonomic coherence: between NCBI and SFE_SE models at all levels
- FD soil/sediment data dilutes regardless of k-mer size

## Practical Recommendation
- **Domain-matched training is best**: Train a 5-mer model on data from your
  target environment. Outperforms general-purpose models phylum through family.
- **NCBI_5 is a solid fallback**: Works on any prokaryotic data (spans tree of
  life). Wins at genus/species due to reference-quality labels.
- **Mixing data sources hurts, but cause unclear**: Adding FD to marine data
  degrades performance. Could be environment mismatch or quality mismatch
  (different sequencing/assembly pipelines, potentially higher error rates).

### Chopped NCBI data (Run_NCBI_5mer_chopped)
Training on NCBI contigs fragmented into >= 100 kbp pieces (1.33M rows from
chop_and_count). Hurt performance: Spearman 0.744, taxonomic coherence 2-3 pp
below un-chopped at every level. Fragments from same genome have similar
profiles — overweights long genomes without adding diversity. The within-genome
variation (genomic islands, HGT, transposons) works against the goal of
mapping the same organism to the same latent region.

### Dropout (Run_NCBI_5mer_drop10)
10% dropout in encoder and decoder hidden layers. Improves taxonomic coherence
at every level (3-6% improvement). Now beats SFE_SE_5 at family, genus, and
species while being competitive at coarser levels. Best overall model.

| Level | No dropout | 10% dropout | SFE_SE_5 |
|---|---|---|---|
| Domain | 0.993 | **0.997** | 0.996 |
| Phylum | 0.908 | **0.938** | **0.955** |
| Class | 0.866 | **0.897** | **0.913** |
| Order | 0.771 | **0.803** | **0.815** |
| Family | 0.686 | **0.711** | 0.705 |
| Genus | 0.515 | **0.534** | 0.504 |
| Species | 0.103 | **0.109** | 0.077 |

Dropout signature: train loss (9.73) > val loss (8.70) because dropout is
active during training only. KL drops from 38 to 33 — dropout provides some
regularization that KL was previously handling.

## Run History
- **Run_NCBI_5mer**: NCBI_5 data, 1000 epochs, 10 min. Final val_loss
  8.36, KL 38. Spearman 0.772.
- **Run_SFE_SE_5mer**: SFE_SE_5 data, 1000 epochs, ~1 hr. Final val_loss
  15.80, KL 94.3. Spearman 0.828.
- **Run_5_5mer**: Full augmented (FD+NCBI+SFE+SE), 1000 epochs, ~2.5 hrs.
  Final val_loss 12.14, KL 68.2. Spearman 0.709.
- **Run_NCBI_5mer_chopped**: NCBI_5 chopped (1.33M rows), 1000 epochs.
  Val_loss 6.16, KL 36.8. Spearman 0.744.
- **Run_NCBI_5mer_drop10**: NCBI_5 with 10% dropout, 1000 epochs.
  Val_loss 8.70, KL 33.2. Spearman 0.771. Best overall taxonomic coherence.
