# VAE_noGC Development Log

## 2026-02-17: Created VAE_noGC.py

Identical to VAE.py except 1-mer features (AT/GC content) are excluded from the input.

### Changes from VAE.py
- `INPUT_DIM` = 2770 (was 2772)
- `COL_END` = 2771 (was 2773) — loads columns 1–2770, skipping 1-mers at cols 2771–2772
- `KMER_SIZES` dictionary has no `'1mer'` entry
- Architecture unchanged: 2770 → 1024 → 512 → 384 latent → 512 → 1024 → 2770

### Motivation
The standard VAE reconstructs GC content (1-mers) with very low error (~0.001 MSE), which means the embedding space partially encodes GC. When clustering, this allows transitivity chains to form along GC gradients — sequences can be linked through intermediate GC values even if their k-mer profiles are otherwise very different.

By removing 1-mers from the input, the VAE must encode sequence composition information without explicit GC content. The question is whether this produces an embedding space where clustering is more biologically meaningful (less GC-driven chaining) or whether the higher-order k-mers already implicitly encode GC and the effect is minimal.
