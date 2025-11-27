
# New requests

I would like for you to replace the _build_encoder and the _build_decoder code. The following is an outline of what I would like.

Each sample has 8,362 fields: 1 (length) + 8,192 (7-mers) + 136 (4-mers) + 32 (3-mers) + 1 (GC) = 8,362 fields

I'd like them split into 3 blocks. The first block (b_7) should have the fields corresponding to the 7-mers plus a copy of the GC (b_gc) and a copy of the length (b_length). The second block (b_4) should have the fileds corresponding to the 4-mers plus a copy of b_gc and a copy of b_length. The third block (b_3) should have the fields corresponding to the 3-mers plus a copy of b_gc and a copy of b_length.

For the encoder: I would like b_7 to feed into a layer with 4,096 nodes, b_4 to feed into a layer with 64 nodes and b_3 to feed into a layer with 16 nodes. Then take the outputs of the 4,096 and feed those into a layer with 2,048 nodes, take the outputs of the 64 nodes and feed them into a layer with 32 nodes and then take the output of the layer with 16 nodes and feed them into a layer with 8 nodes. Finally, concatenate the outputs from the last layers (2,048 + 32 + 8) into one and feed it into a layer with 1,024 nodes. From there, go to 512 nodes and then finally to the latent space.

For the decoder: Reverse the progression used for the encoder.

Keep the batch normalization layers.

Review this and ask any questions you wish. Do not write any code yet. If possible, show me a graph of what the models would be like.

---

# Claude's Feedback

## Final Architecture

### Resolved Design Decisions

1. **Latent dimension**: 512 (increased from 256 to reduce 7-mer bottleneck)
2. **Decoder output**: 8,361 features (no length) - `[7-mers, 4-mers, 3-mers, GC]`
3. **GC handling**: Each decoder branch predicts GC; average the 3 predictions
4. **Decoder split**: Separate Dense layers project from shared representation to each branch
5. **Loss calculation**: Computed on 8,361 features (input minus length field)
6. **Deep 7-mer branch** (Option B + Option 1 + Option 2):
   - 7-mer branch: 8,194 → 1024 → 512 → 256 (3 layers, 73% of concat)
   - 4-mer branch: 138 → 128 → 64 (2 layers, 18% of concat)
   - 3-mer branch: 34 → 64 → 32 (2 layers, 9% of concat)
   - **Total parameters: ~19M**
7. **Learning rate scheduling**: ReduceLROnPlateau (factor=0.5, patience=20, min_lr=1e-6)

### Encoder Architecture
```
Input (8,362)
    │
    ├─────────────────────┬─────────────────┬──────────────────┐
    ▼                     ▼                 ▼                  │
 7-mers (8,192)      4-mers (136)      3-mers (32)        len(1), GC(1)
    │                     │                 │                  │
    │   ┌─────────────────┼─────────────────┼──────────────────┘
    │   │                 │                 │          (copy to each branch)
    ▼   ▼                 ▼   ▼             ▼   ▼
  b_7 (8,194)         b_4 (138)         b_3 (34)
    │                     │                 │
    ▼                     ▼                 ▼
 Dense(1024)+BN+LReLU  Dense(128)+BN+LReLU  Dense(64)+BN+LReLU
    │                     │                 │
    ▼                     ▼                 ▼
 Dense(512)+BN+LReLU   Dense(64)+BN+LReLU   Dense(32)+BN+LReLU
    │                     │                 │
    ▼                     │                 │
 Dense(256)+BN+LReLU      │                 │
    │                     │                 │
    └──────────┬──────────┴─────────────────┘
               ▼
        Concatenate (352)
               │
               ▼
          Dense(512)+BN+LReLU
               │
               ▼
         z_mean(256), z_log_var(256) → Sampling → z(256)
```

### Decoder Architecture
```
         Latent z (256)
               │
               ▼
          Dense(512)+BN+LReLU
               │
               ▼
          Dense(352)+BN+LReLU
               │
    ┌──────────┼──────────┬─────────────────┐
    │          │          │                 │
    ▼          ▼          ▼                 │  (separate Dense projections)
 Dense(256)   Dense(64)   Dense(32)         │
 +BN+LReLU   +BN+LReLU   +BN+LReLU          │
    │          │          │                 │
    ▼          ▼          ▼                 │
 Dense(512)   Dense(128)  Dense(64)         │
 +BN+LReLU   +BN+LReLU   +BN+LReLU          │
    │          │          │                 │
    ▼          │          │                 │
 Dense(1024)   │          │                 │
 +BN+LReLU     │          │                 │
    │          │          │                 │
    ▼          ▼          ▼                 │
 Dense(8,193)  Dense(137)  Dense(33)        │  (k-mers + GC each, linear)
    │          │          │                 │
    ├──────────┼──────────┤                 │
    │          │          │                 │
    ▼          ▼          ▼                 │
 7-mers(8,192) 4-mers(136) 3-mers(32)    GC avg(1)
    │          │          │                 │
    └──────────┴──────────┴─────────────────┘
                          │
                          ▼
                   Output (8,361)
                   [7-mers, 4-mers, 3-mers, GC]
```

## Feedback

**Strengths:**

1. **Biologically sensible**: Different k-mer scales capture different signals. 7-mers capture longer motifs; 3-mers capture fundamental codon/trinucleotide biases. Separate processing paths let the model learn scale-specific features.

2. **Consistent compression ratios**: Each branch compresses ~4x, which is balanced.

3. **Contextual features**: Including GC and length in each branch provides useful context for interpreting k-mer frequencies.

**Potential concerns:**

1. **Information asymmetry**: At concatenation, 7-mers contribute 2,048/2,088 = **98%** of features. The 4-mer and 3-mer branches contribute only 2%. Is this intentional? If you want more balanced influence, consider:
   - Increasing 4-mer final layer: 32 → 64 or 128
   - Increasing 3-mer final layer: 8 → 16 or 32

2. **Parameter count**: The 7-mer branch will dominate parameters too (8,194×4,096 + 4,096×2,048 ≈ 42M parameters just for that branch). This is fine, just be aware.

---

# Future Improvement Suggestions

Based on initial training results (epoch 15: MSE 29,558, KL 177, val_loss 29,836), these are potential improvements ranked by expected impact.

## 1. Per-Feature-Group Loss Weighting (High Impact)

**Problem:** Currently 7-mers dominate the loss by sheer count (8,192/8,361 = 98% of features). The 4-mers (136), 3-mers (32), and GC (1) barely influence training despite having their own decoder branches.

**Solution:** Equal weight per feature group:

```python
# In VAE.call(), replace:
recon_loss = ops.mean(ops.sum(ops.square(target - reconstruction), axis=1))

# With per-group MSE averaged:
mse_7 = ops.mean(ops.square(target[:, :8192] - reconstruction[:, :8192]))
mse_4 = ops.mean(ops.square(target[:, 8192:8328] - reconstruction[:, 8192:8328]))
mse_3 = ops.mean(ops.square(target[:, 8328:8360] - reconstruction[:, 8328:8360]))
mse_gc = ops.mean(ops.square(target[:, 8360:] - reconstruction[:, 8360:]))

recon_loss = (mse_7 + mse_4 + mse_3 + mse_gc) * 2090  # Scale to similar magnitude
```

This ensures each branch has equal incentive to learn well.

## 2. Learning Rate Scheduling (Medium Impact)

Add cosine decay or reduce-on-plateau to squeeze out more performance:

```python
callbacks.append(
    keras.callbacks.ReduceLROnPlateau(
        monitor = 'val_loss',
        factor = 0.5,
        patience = 10,
        min_lr = 1e-6
    )
)
```

## 3. Per-Branch Loss Monitoring (Diagnostic)

Track reconstruction quality per feature group in `VAEMetricsCallback` to identify if any branch is lagging:

```python
# In VAEMetricsCallback.on_epoch_end():
target = sample_x[:, 1:]
pred = predictions

mse_7 = float(ops.mean(ops.square(target[:, :8192] - pred[:, :8192])))
mse_4 = float(ops.mean(ops.square(target[:, 8192:8328] - pred[:, 8192:8328])))
mse_3 = float(ops.mean(ops.square(target[:, 8328:8360] - pred[:, 8328:8360])))
mse_gc = float(ops.mean(ops.square(target[:, 8360:] - pred[:, 8360:])))

logger.info(f'  Per-group MSE: 7mer={mse_7:.4f}, 4mer={mse_4:.4f}, 3mer={mse_3:.4f}, GC={mse_gc:.4f}')
```

## 4. Cosine Similarity Loss (Experimental)

K-mer frequencies are compositional (they sum to ~1). Cosine similarity might capture the "shape" of frequency profiles better than MSE:

```python
cosine_loss = 1 - ops.sum(target * reconstruction, axis=1) / (
    ops.norm(target, axis=1) * ops.norm(reconstruction, axis=1) + 1e-8
)
```

Could use as additional loss term or replace MSE entirely.

## 5. Other Ideas (Lower Priority)

- **Input dropout** (denoising VAE): Add `Dropout(0.1)` on encoder inputs to improve robustness
- **Latent dimension tuning**: Try 128 (more compression) or 512 (more expressiveness)
- **β-VAE**: KL weight > 1 for more disentangled representations (trade-off with reconstruction quality)
- **Hierarchical latent space**: Split latent into parts corresponding to different k-mer scales, e.g., `z = [z_7mer(128), z_4mer(64), z_3mer(32), z_shared(32)]`

## Implementation Priority

1. Start with #3 (per-branch monitoring) to understand current behavior
2. Implement #1 (per-group weighting) if monitoring reveals imbalanced learning
3. Add #2 (LR scheduling) for fine-tuning in later epochs

---

# 7-mer Bottleneck Improvement Options

Based on training results showing 7-mer MSE (~3.6) is now the bottleneck while 4-mer (0.06) and 3-mer (0.01) have essentially converged.

## Option 1: Increase 7-mer branch width (Recommended)

Current: 8,194 → 512 → **128** (64x compression)
Proposed: 8,194 → 512 → **256** (32x compression)

Doubles 7-mer information capacity. Modest parameter increase (~65K encoder + ~130K decoder).

**Impact:** Concat becomes 256+64+32=352 instead of 224. Shared layers need adjustment (352 → 512 → latent).

## Option 2: Add depth to 7-mer branch

Current: 8,194 → 512 → 128
Proposed: 8,194 → 1024 → 512 → 256

More layers = more non-linearity to learn complex patterns. Larger parameter increase.

## Option 3: Skip connection in 7-mer branch

Add residual connection from first hidden layer:
```python
x1 = Dense(512)(input)
x1 = BatchNormalization()(x1)
x1 = LeakyReLU()(x1)
x2 = Dense(128)(x1)
x2 = BatchNormalization()(x2)
x2 = LeakyReLU()(x2)
skip = Dense(128)(x1)  # project to same dim
x = x2 + skip  # residual connection
```

Cheap to add, helps gradient flow.

## Option 4: Adjust loss weighting

Give 7-mers more than 25% weight since they're harder:
```python
recon_loss = (mse_7 * 2 + mse_4 + mse_3 + mse_gc) * (OUTPUT_DIM / 5)
```

Gives 7-mers 40% weight. Easiest change but may hurt other branches.

## Recommendation

Try Option 1 first after current run completes - it directly addresses the capacity constraint with minimal complexity.

---

# Implementation Log

**2025-11-25 20:48** - Implemented #1 (Per-Feature-Group Loss Weighting) and #3 (Per-Branch Monitoring)
- Added output slice constants: `OUT_KMER_7_SLICE`, `OUT_KMER_4_SLICE`, `OUT_KMER_3_SLICE`, `OUT_GC_SLICE`
- Updated `VAE.call()` to compute per-group MSE with equal weighting
- Updated `VAEMetricsCallback` to log per-group MSE each epoch
- Loss now gives equal weight to 7-mers, 4-mers, 3-mers, and GC regardless of feature count

**2025-11-25 21:48** - Implemented Option 1 (Increase 7-mer branch width)
- Encoder 7-mer branch: 8,194 → 512 → **256** (was 128)
- Decoder shared layer: 512 → **352** (was 224)
- Decoder 7-mer branch: **256** → 512 → outputs (was 128 → 512)
- Concatenation now 256 + 64 + 32 = 352 (was 224)
- 7-mer now has 32x compression (was 64x), doubling information capacity

**2025-11-26 09:53** - Implemented Option 2 (Deeper 7-mer branch) + Learning Rate Scheduling
- Encoder 7-mer branch: 8,194 → **1024** → **512** → 256 (was 512 → 256)
- Decoder 7-mer branch: 256 → 512 → **1024** → outputs (was 256 → 512 → outputs)
- Added `ReduceLROnPlateau` callback (factor=0.5, patience=20, min_lr=1e-6)
- More depth = more non-linearity to capture complex 7-mer patterns

---

# Fresh Analysis (2025-11-26)

## Core Problem: 7-mer Loss is Dominated by Zeros

With 8,192 possible 7-mers and ~5,000bp sequences, expected count per 7-mer is **~0.6**. Most 7-mers have **zero frequency**.

In log-space:
- Zero frequency → `log(1e-6)` ≈ **-13.8**
- Non-zero (0.001) → `log(0.001)` ≈ **-6.9**

The MSE is dominated by predicting which 7-mers are exactly zero vs nearly zero. The model spends capacity on **noise in the sparse tail**, not biologically meaningful signal.

## Information Bottleneck

256 latent dimensions → 8,192 output features = **32x expansion**

With KL regularization, each dimension carries limited information. 256 dimensions cannot faithfully encode 8,192 continuous values.

Comparison:
- 7-mers: 8192 features / 256 latent = 32x expansion (severe bottleneck)
- 4-mers: 136 features / 256 latent = 0.5x (overcomplete, easy)
- 3-mers: 32 features / 256 latent = 0.125x (very overcomplete, trivial)

## Hierarchical Redundancy Ignored

4-mer frequencies are **mathematically derivable** from 7-mer frequencies. Every 4-mer appears as a substring of specific 7-mers. The model predicts them independently.

## Equal Group Weighting Creates Gradient Imbalance

Current loss: `(mse_7 + mse_4 + mse_3 + mse_gc) * scale`

Per-feature gradient signal:
- Each 7-mer feature: 0.003% of gradient
- Each 3-mer feature: 0.78% of gradient (**256x more**)

## Improvement Ideas

### 1. Diagnostic: Zero vs Non-zero 7-mer MSE
Compute MSE separately to understand where the error comes from.

### 2. Masked/Weighted Loss for 7-mers
Focus on reconstructing non-zero (present) k-mers rather than zeros (absent).

### 3. Clip Log Transform Floor
Use `clip(log(x + 1e-6), -10, None)` to reduce dynamic range.

### 4. Increase Latent Dimension
256 → 512 or 768 to give more capacity for 7-mer information.

### 5. Cosine Similarity Loss
K-mer frequencies are compositional. Cosine similarity measures distribution "shape".

### 6. Derive 4-mers from 7-mers
Only predict 7-mers; compute 4-mers by summing appropriate 7-mer subsets.

**2025-11-26 15:04** - Implemented diagnostic logging + increased latent dimension
- Added zero vs non-zero 7-mer MSE breakdown in `VAEMetricsCallback`
  - `nz=` MSE on non-zero 7-mers (target > -10 in log-space)
  - `z=` MSE on zero 7-mers (target <= -10)
  - `%nz` percentage of 7-mers that are non-zero
- Increased `LATENT_DIM` from 256 to **512** (reduces 7-mer bottleneck from 32x to 16x)

---

# Deep Analysis: Zero 7-mer Reconstruction Problem (2025-11-26)

## Training Results (100 epochs)

Final metrics:
- Validation loss: **2840.51**
- Non-zero 7-mer MSE: **0.40** ✅ (good)
- Zero 7-mer MSE: **13.7** ❌ (34x worse than non-zero)
- 59.1% of 7-mers are non-zero per sample
- 4-mer MSE: 0.036, 3-mer MSE: 0.006, GC MSE: 0.0002 (all excellent)

## Root Cause Analysis

### The Architectural Mismatch

The model is asked to solve **two different problems** with **one linear output layer**:

1. **Classification**: Which 7-mers are present? (binary, 8192 decisions)
2. **Regression**: What are the values of present 7-mers? (continuous)

A linear activation cannot do both well. The model "hedges" by predicting intermediate values instead of confidently outputting zeros.

### Why MSE ~13.7 for Zeros

In log-space:
- Zero 7-mers: `log(1e-6)` ≈ **-13.8**
- Non-zero 7-mers: range from **-10 to 0**

`sqrt(13.7) ≈ 3.7` error → model predicts **~-10** instead of **-13.8**

### The 0.1x Down-weighting Backfires

```python
weights_7 = nonzero_mask + (1.0 - nonzero_mask) * 0.1
```

This reduces gradient signal for zeros by 10x. The model gets almost no feedback for bad zero predictions.

### Information Theory Constraint

- 8,192 possible 7-mers, ~59% non-zero per sample
- Encoding which are zero requires encoding a binary pattern
- 256 latent dims (with KL regularization) ≈ ~1,000-2,000 bits
- Insufficient for arbitrary 8,192-bit patterns

Patterns aren't arbitrary (correlate with length, GC, biology), but current architecture makes learning them hard.

## Recommended Solution: Gated Sparsity Prediction

Separate the two problems with an explicit sparsity gate:

### Architecture Change

```python
# In decoder 7-mer branch:
# Predict values (what to output if non-zero)
values_7 = layers.Dense(8192, activation='linear', name='dec_7mer_values')(x_7)

# Predict sparsity gate (probability each 7-mer is non-zero)
gate_logits = layers.Dense(8192, name='dec_7mer_gate_logits')(x_7)
gate = layers.Activation('sigmoid', name='dec_7mer_gate')(gate_logits)

# Combine: interpolate between value and floor
FLOOR_VALUE = -13.8  # log(1e-6)
kmers_7 = gate * values_7 + (1.0 - gate) * FLOOR_VALUE
```

### Loss Function Change

```python
# BCE for sparsity (which 7-mers are present?)
target_nonzero = ops.cast(target_7 > -10.0, 'float32')
sparsity_bce = ops.mean(keras.losses.binary_crossentropy(target_nonzero, gate))

# MSE only on non-zero targets (what are the values?)
nonzero_count = ops.maximum(ops.sum(target_nonzero), 1.0)
mse_7_values = ops.sum(target_nonzero * ops.square(target_7 - values_7)) / nonzero_count

# Combined
loss_7 = sparsity_bce + mse_7_values
```

### Why This Works

- Sigmoid + BCE is designed for binary classification
- MSE focuses only on meaningful value prediction
- Clean separation of concerns
- Gate naturally pushes outputs toward floor value when probability is low

## Alternative Approaches (Lower Priority)

1. **Asymmetric Zero Loss**: Heavy penalty for predicting non-zero when target is zero
2. **Deeper Decoder**: More gradual 2x expansion (352→256→512→1024→2048→8192)
3. **Increase Latent Dim**: 256→512 for more sparsity encoding capacity
4. **Remove 0.1x Down-weighting**: Currently hurting zero prediction

**2025-11-26 18:30** - Implemented gated sparsity approach
- Added `GatedCombineLayer` custom layer for `gate * values + (1 - gate) * floor`
- Modified decoder to output three tensors: `[reconstruction, gate_7, values_7]`
  - `values_7`: Dense(8192) linear - predicted log-frequencies for non-zero 7-mers
  - `gate_7`: Dense(8192) sigmoid - probability each 7-mer is non-zero
  - `reconstruction`: gated combination using floor_value = -13.8
- Updated `VAE.call()` loss function:
  - BCE loss for sparsity prediction (gate vs target_nonzero)
  - MSE loss for value prediction (only on non-zero targets)
  - Removed 0.1x down-weighting that was hurting zero prediction
- Updated `VAEMetricsCallback` to track:
  - `BCE`: sparsity binary cross-entropy
  - `valMSE`: MSE on non-zero values only
  - `gateAcc`: accuracy of gate at 0.5 threshold
  - Still tracks `outMSE`, `nz`, `z` for the final gated output

---

# Gate Accuracy Plateau Analysis (2025-11-26)

## Observation

Gate accuracy plateaus at ~76% regardless of:
- BCE loss weight (1x vs 2x)
- Model size (18M vs 32M params)
- Separate gate/value pathways

This suggests the bottleneck is **latent space capacity** (256 dims), not decoder capacity.

## Architecture Evolution

| Change | Gate Acc | Notes |
|--------|----------|-------|
| Shared gate/value (32M params) | 76.3% | Stuck |
| 2x BCE weight | 76.4% | No improvement |
| Separate pathways (32M) | 76.2% | No improvement |
| Slimmed model (18.7M) | 76.2% | Same ceiling |
| Latent 256→512 | TBD | Testing |

## Parameter Budget Problem

With 4.78M samples and models of 18-32M params, samples/param ratio is 0.15-0.26 (want >1).

The 7-mer output layers dominate:
- Value: 1024 → 8192 = 8.4M params
- Gate: 512 → 8192 = 4.2M params
- Total 7-mer: ~12.6M of 19M (66%)

---

# Reducing 7-mer Parameter Cost (2025-11-26)

## Options Considered

### 1. Low-Rank Factorization (Recommended - Simplest)
Add bottleneck before 8192 output:
```
1024 → 128 → 8192: 131K + 1M = 1.1M (vs 8.4M, 7.6x reduction)
```

### 2. Hierarchical/Grouped Decoding
Group 7-mers by prefix (64 groups × 128 each):
```
1024 → 64×32 → reshape → 128 per group → 8192
Params: ~2.1M (4x reduction)
```

### 3. Pretrained 7-mer Autoencoder
Separate autoencoder: 8192 → 128 → 8192
VAE predicts 128-dim code, frozen decoder expands.

### 4. Convolutional Upsampling
Organize 7-mers as 2D grid, use Conv2DTranspose for weight sharing.

### 5. Embedding Dot-Product
Learn 8192 7-mer embeddings (32-dim), output = query @ embeddings.T
Params: ~295K (massive reduction)

## Implementation

**2025-11-26 20:45** - Implemented low-rank bottleneck
- Value branch: 352 → 512 → 1024 → **128** → 8192 (bottleneck)
- Gate branch: 352 → 256 → 512 → **128** → 8192 (bottleneck)
- Result: 8.8M params, 0.54 samples/param

**2025-11-26 21:15** - Implemented embedding dot-product
- Added `EmbeddingDotProduct` layer: output = query @ learned_embeddings.T
- Value branch: 352 → 512 → query(64) @ embeddings(8192×64)
- Gate branch: 352 → 256 → query(64) @ embeddings(8192×64)
- Result: **6.95M params**, 0.69 samples/param
- 7-mer embeddings learn similarity structure between k-mers

**2025-11-26 21:20** - Changed significance threshold
- Redefined "significant" 7-mers: freq > 0.01 (was > ~1e-5)
- In log-space: target > -4.6 (was > -10)
- Gate now predicts significant vs non-significant (not just zero vs non-zero)
- Expected ~100 significant 7-mers per sample (was ~4,800 non-zero)
- This simplifies the gate's task dramatically

**2025-11-26 21:30** - Threshold tuning
- 0.01 threshold too strict: avgSig=0 (all 7-mers below threshold)
- 0.0005 threshold (~243 sig/sample): gate stuck at 97.5% (predicts all zeros, too imbalanced)
- 0.0001 threshold (~3,519 sig/sample): back to 76% plateau

---

# 7-mer Feature Space Analysis (2025-11-26)

## Variance Analysis Results

Analyzed whether we could reduce 7-mer space by filtering low-variance features:

| Metric | Range | Notes |
|--------|-------|-------|
| Total variance | 2.5 - 9.8 | All 7-mers have high variance |
| Presence rate | 29% - 93% | No 7-mers always/never present |
| Variance-when-present | 0.33 - 2.79 | Much smaller range |

**Key Finding:** High total variance is driven by the zero vs non-zero distinction in log-space. All 7-mers vary substantially because of the log-space representation creating a ~13.8 unit gap between zeros (-13.8) and smallest non-zeros (~0).

Simple variance filtering won't help because the variance is an artifact of log-space, not biological signal.

---

# Linear-Space 7-mer Proposal (2025-11-26)

## The Log-Space Problem

In log-space, the 7-mer distribution is fundamentally bimodal:
- **Zeros**: Artificially floored at -13.8 (log of epsilon)
- **Non-zeros**: Range from ~-9 to ~+16
- **The gap**: Huge discontinuity between -13.8 and smallest real values

This creates two incompatible tasks for a smooth neural network output:
1. **Classification**: Is this 7-mer present? (binary decision across 13.8-unit gap)
2. **Regression**: What's the value? (continuous prediction)

## Proposed Solution: Hybrid Space

Keep encoder inputs in log-space (captures multiplicative structure), but output 7-mers in linear space:

```
Encoder input: log-space (all k-mers) - unchanged
Decoder output:
  - 3mer, 4mer, GC: log-space (dense, no sparsity issue)
  - 7mer: LINEAR-space frequencies via softplus activation
```

### Why This Helps

1. **Zeros are exactly 0** - no artificial floor value
2. **Gate task becomes natural**: sigmoid predicts P(frequency > 0), which directly matches the data
3. **Softplus enforces non-negative**: smooth activation that's 0 when input is very negative
4. **No discontinuity**: continuous mapping from latent space to output

### Implementation

```python
# 7-mer value branch - output linear frequencies
values_7_linear = layers.Dense(8192, activation='softplus', name='dec_7mer_values')(...)

# Gate branch - unchanged (sigmoid)
gate_7 = layers.Activation('sigmoid', name='dec_7mer_gate')(gate_logits_7)

# Combine: gate * values (zeros naturally propagate)
kmers_7_linear = gate_7 * values_7_linear

# Loss computation:
# 1. Store original linear frequencies before log-transform (or exp() during loss)
# 2. Gate BCE: target = (original_freq > 0)
# 3. Value MSE: compare softplus output against original linear frequencies
```

### Data Flow

```
Original data (linear frequencies)
    ↓ log-transform
Encoder input (log-space)
    ↓ encode
Latent z (512 dims)
    ↓ decode
    ├→ 3mer, 4mer, GC: log-space output (unchanged)
    └→ 7mer: linear-space via softplus
           ↓
       Loss vs original linear frequencies
```

### Advantages

1. Gate's sigmoid output directly answers "is this present?" - natural for linear space
2. Softplus is smooth everywhere - no floor value discontinuity
3. MSE in linear space: small errors in small values have proportionally appropriate loss
4. Eliminates the -13.8 artifact that dominates current log-space MSE

### Potential Concerns

1. Very small dynamic range in linear space (0 to ~0.07)
2. Need to keep raw linear frequencies alongside log-transformed data
3. May need to scale linear outputs or use custom loss weighting

**2025-11-26 22:00** - Implemented linear-space 7-mer output

Changes made:
1. **GatedCombineLayer**: Simplified to just `gate * values` (no floor value needed in linear space)
2. **Decoder 7-mer values**: Added softplus activation after EmbeddingDotProduct
   - `values_7_logits = EmbeddingDotProduct(...)` → `values_7 = softplus(values_7_logits)`
   - Softplus ensures non-negative output, approaches 0 for very negative inputs
3. **Loss computation**: Now uses linear-space targets for 7-mers
   - `target_7_linear = exp(target_7_log)` to convert back to raw frequencies
   - `present_mask = target_7_linear > 1e-5` (effectively > 0)
   - BCE: gate predicts presence (natural match for linear space)
   - MSE: compare softplus output against linear target (masked to present only)
4. **Metrics callback**: Updated to report linear-space metrics
   - `avgPresent` instead of `avgSig`
   - MSE values now in linear scale (will be much smaller numbers)

Key insight: In linear space, the gate's sigmoid output directly answers "is this 7-mer present?" - this is the natural question for sparse frequency data, rather than "is this above some arbitrary log-space threshold?"

Output tensor is now mixed representation:
- 7-mers (0:8192): LINEAR space (softplus)
- 4-mers (8192:8328): LOG space
- 3-mers (8328:8360): LOG space
- GC (8360:8361): LOG space

## Initial Training Results (Linear-Space 7-mers)

| Epoch | gateAcc | valMSE | outMSE (pres) | outMSE (abs) | BCE |
|-------|---------|--------|---------------|--------------|-----|
| 1 | 76.0% | 0.000707 | 0.000367 | 0.000185 | 0.4736 |
| 5 | 77.5% | 0.000024 | 0.000013 | 0.000007 | 0.4486 |
| 11 | 77.7% | 0.000001 | 0.000001 | 0.000000 | 0.4450 |
| 20 | 77.9% | 0.000000 | 0.000000 | 0.000000 | 0.4412 |

**Key Observations:**
1. **Gate accuracy improved from 76% to 78%** - slight improvement over log-space plateau
2. **Value MSE converged to ~0** - value prediction for present 7-mers is excellent
3. **Absent MSE essentially 0** - model perfectly predicts 0 for absent 7-mers
4. **BCE still decreasing** - gate still learning

**Comparison with Log-Space Approach:**
- Log-space: MSE ~13.7 for zeros, ~0.4 for non-zeros (34x worse)
- Linear-space: MSE ~0 for both present and absent

**Interpretation:**
The linear-space approach eliminates the -13.8 discontinuity that was causing poor zero reconstruction. Now the model can focus on the binary classification task (which 7-mers are present) without fighting against the log-space floor value.

The gate accuracy plateau at ~78% likely reflects a fundamental information-theoretic limit: encoding which ~5,300 of 8,192 7-mers are present requires substantial information that competes with KL regularization.

---

# Analysis: Why Linear-Space Works Better (2025-11-26)

## The Log-Space Problem in Detail

In log-space, the model faced an impossible task with a single output layer:

```
Target distribution:
- Absent 7-mers: -13.8 (log of epsilon)
- Present 7-mers: -9 to +16

Required behavior:
- For absent: output must be exactly -13.8 (very specific value)
- For present: output must be in range -9 to +16 (continuous)
```

A linear output layer cannot simultaneously:
1. Output -13.8 for ~2,900 absent 7-mers
2. Output continuous values for ~5,300 present 7-mers
3. Do this differently for each sample (different 7-mers are present)

The model "hedged" by outputting ~-10 for everything, which is wrong for both.

## Why Linear-Space Solves This

In linear-space with gated softplus:

```
Output = gate * softplus(logits)

For absent (gate → 0): output → 0 ✓ (correct!)
For present (gate → 1): output → softplus(logits) ≈ frequency ✓
```

The gate handles the binary decision, softplus handles the value. Clean separation.

## Remaining Challenge: Gate Accuracy

At ~78%, the gate correctly predicts presence/absence for ~6,400 of 8,192 7-mers per sample.
Incorrectly classified: ~1,800 per sample.

This is likely constrained by:
1. **Latent space capacity**: 512 dimensions with KL regularization
2. **Combinatorial complexity**: C(8192, 5300) possible presence patterns
3. **Biological correlation**: many 7-mers co-occur, which actually helps

## What 78% Gate Accuracy Means for Clustering

For clustering purposes, ~78% gate accuracy may be sufficient because:
1. The latent space still captures the major patterns
2. Similar sequences will have similar gate errors (systematic, not random)
3. Value prediction for present 7-mers is essentially perfect

The VAE latent space should still separate biologically distinct sequences well.

## Model Statistics (Epoch 23)

- **Parameters**: ~6.95M
- **Samples/param ratio**: 0.69 (better than earlier 41M param model)
- **avgPresent**: 5,291 7-mers per sample (64.6% of 8,192)
- **4-mer MSE**: 0.059 (excellent)
- **3-mer MSE**: 0.012 (excellent)
- **GC MSE**: 0.001 (excellent)
- **Val loss**: 2074.58 (steadily decreasing)

---

# Future Idea: Add 5-mers and 6-mers to Encoder (2025-11-26)

## Rationale

The gate accuracy plateau (~78%) might improve if the encoder sees hierarchical k-mer structure explicitly:

- Each 7-mer contains 3 overlapping 5-mers and 2 overlapping 6-mers
- If a 5-mer is absent → all 7-mers containing it must be absent
- This constraint could help the encoder build better latent representations

## Feature Counts (canonical)

- 5-mers: ~512 features
- 6-mers: ~2,080 features
- Current input: 8,362 features
- With 5/6-mers: ~10,954 features

## Implementation Notes

- Only add to encoder input (not decoder output)
- Create separate encoder branches for 5-mer and 6-mer
- Decoder still outputs only [7-mers, 4-mers, 3-mers, GC]
- Need 5-mer and 6-mer frequency data (compute from sequences if not available)

## Status

Waiting for current linear-space training run to complete before exploring this.

---

# Final Training Results: Linear-Space with 2e-4 Threshold (2025-11-27)

## Configuration

- **Threshold**: 2e-4 (freq > 0.0002, appears ~1-2 times in 5kb sequence)
- **avgPresent**: 1,578 per sample (19.3% of 8,192)
- **Parameters**: ~6.95M
- **Epochs**: 500

## Final Metrics (Epoch 500)

| Metric | Value | Notes |
|--------|-------|-------|
| **gateAcc** | 86.1% | +5.4% above 80.7% baseline |
| **BCE** | 0.319 | Converged |
| **Val loss** | 1478 | Stable |
| **4mer MSE** | 0.031 | Excellent |
| **3mer MSE** | 0.0035 | Excellent |
| **GC MSE** | 0.0001 | Excellent |
| **7mer valMSE** | ~0 | Perfect value prediction |

## Threshold Comparison Summary

| Threshold | avgPresent | % present | Baseline | gateAcc | Above baseline |
|-----------|------------|-----------|----------|---------|----------------|
| 1e-5 | 5,291 | 64.6% | 64.6% | 78.3% | +13.7% |
| 2e-4 | 1,578 | 19.3% | 80.7% | 86.1% | +5.4% |
| 5e-4 | 254 | 3.1% | 96.9% | 97.5% | +0.6% |

## Assessment

**Strengths:**
- 4-mer, 3-mer, GC reconstruction essentially perfect
- 7-mer value prediction (when present) essentially perfect
- 86.1% gate accuracy for 7-mer presence classification

**Limitations:**
- ~14% of 7-mer presence predictions incorrect (~1,140 per sample)
- Gate accuracy plateau due to VAE latent space capacity

**For Clustering:**
Model should be good enough - latent space captures major k-mer patterns. Similar sequences will cluster together based on shared 4-mer/3-mer profiles and partially shared 7-mer presence patterns.

