# VAE Request

I would like you to build a new VAE. You can edit the old VAE.py file as a skeleton.

As a first cut, use just the 7-mers from Data/all_mkers.npy.

I would like you to build the VAE starting with two 1D convolutional layers and one fully connected layer before going to a 256-dimensional latent space. 

Ask me any questions you need to and put a copy of the questions in this file as well as in the terminal. I will add my answers here or in the terminal. You should keep a running log of your decisions - data and time stamped - in this file.

---

## 2025-12-01 Questions from Claude

1. **Conv1D input shape**: The 7-mer data has 8,192 features. For 1D convolution, should I treat this as a 1D signal of length 8192 with 1 channel, or reshape it differently (e.g., 128×64)?

2. **Kernel sizes**: What kernel sizes would you like for the two conv layers? Common choices are 3, 5, 7, or larger for capturing longer-range patterns.

3. **Number of filters**: How many filters per conv layer? (e.g., 64, 128, 256)

4. **Pooling**: Should I use pooling (max or average) between/after conv layers to reduce dimensionality before the fully connected layer?

5. **Decoder architecture**: Should the decoder mirror the encoder (transposed convolutions / upsampling), or use a simpler fully-connected decoder?

6. **Loss function**: The old VAE used BCE for 6-mers. For 7-mers, would you prefer:
   - BCE (treats frequencies as independent probabilities)
   - MSE (standard regression)
   - MSE in log-space (better for small values)
   - Something else?

## Answers from Torben

First of all, please remember to add a *timestamp* as well as a date to your entries here. It can be approximate.

- Treat it as a 1D signal of length 8,192 with 1 channel.

- For the first layer, use a filter length of 7 and 256 filters. For the second, use a filter length of 15 and 512 filters. Put
  a max pooling layer in between.

- Mirror the encoder in the decoder.

- Use BCE for the loss function.

This is an experiment to see what happens if we add in 1D convolutional layers and I fully expect to iterate several times.

---

## 2025-12-01 ~14:30: Design decisions

Based on Torben's answers, the architecture will be:

### Encoder
1. Input: (8192, 1) - 7-mer frequencies as 1D signal
2. Conv1D: 256 filters, kernel size 7, padding='same'
3. MaxPooling1D: pool size 2 → (4096, 256)
4. Conv1D: 512 filters, kernel size 15, padding='same'
5. Flatten → 4096 × 512 = 2,097,152
6. Dense → 256 (latent space)

**Issue**: The flattened size (2M) is very large. Will add a second pooling layer or use global pooling to reduce dimensionality before the dense layer. Need to clarify with Torben.

### Decoder (mirror of encoder)
1. Dense: latent → flattened size
2. Reshape → (pooled_length, 512)
3. Conv1DTranspose: 256 filters, kernel size 15
4. Upsample
5. Conv1DTranspose: 1 filter, kernel size 7
6. Output: (8192, 1) with sigmoid

### Loss
- BCE (binary cross-entropy) for reconstruction

---

## 2025-12-01 ~14:45: Revised architecture

After discussion, revised to reduce parameter count:

### Encoder
```
Input (8192, 1)
    │
    ▼
Conv1D (128 filters, kernel=7, padding='same')
    │
    ▼
MaxPooling1D (pool=4)
    │
    ▼
(2048, 128)
    │
    ▼
Conv1D (256 filters, kernel=15, padding='same')
    │
    ▼
MaxPooling1D (pool=4)
    │
    ▼
(512, 256)
    │
    ▼
Flatten
    │
    ▼
(131,072)
    │
    ▼
Dense (1024)
    │
    ▼
Dense (z_mean: 256)  Dense (z_log_var: 256)
    │                      │
    └──────────┬───────────┘
               ▼
         Sampling (z)
               │
               ▼
           (256) latent
```

### Decoder
```
Input (256) latent
    │
    ▼
Dense (1024)
    │
    ▼
Dense (131,072)
    │
    ▼
Reshape (512, 256)
    │
    ▼
UpSampling1D (size=4)
    │
    ▼
(2048, 256)
    │
    ▼
Conv1DTranspose (128 filters, kernel=15, padding='same')
    │
    ▼
UpSampling1D (size=4)
    │
    ▼
(8192, 128)
    │
    ▼
Conv1DTranspose (1 filter, kernel=7, padding='same', activation='sigmoid')
    │
    ▼
Output (8192, 1)
```

### Parameter count estimate
- Conv1D 1: 7 × 1 × 128 + 128 = 1,024
- Conv1D 2: 15 × 128 × 256 + 256 = 491,776
- Dense (flatten→1024): 131,072 × 1024 + 1024 = 134M
- Dense (1024→256): 1024 × 256 × 2 + 512 = 525K (z_mean + z_log_var)
- Decoder mirrors: ~134M

Total: ~270M parameters (dominated by flatten→dense connection)

---

## 2025-12-01 ~15:00: Final architecture

Revised to get flatten layer down to 8,192 using three pooling layers:

### Encoder
```
Input (8192, 1)
    │
    ▼
Conv1D (64 filters, kernel=7, padding='same')
    │
    ▼
MaxPooling1D (pool=4) → (2048, 64)
    │
    ▼
Conv1D (64 filters, kernel=15, padding='same')
    │
    ▼
MaxPooling1D (pool=4) → (512, 64)
    │
    ▼
MaxPooling1D (pool=4) → (128, 64)
    │
    ▼
Flatten → 8,192
    │
    ▼
Dense (1024)
    │
    ▼
Dense (z_mean: 256)  Dense (z_log_var: 256)
    │                      │
    └──────────┬───────────┘
               ▼
         Sampling (z)
               │
               ▼
           (256) latent
```

### Decoder
```
Input (256) latent
    │
    ▼
Dense (1024)
    │
    ▼
Dense (8,192)
    │
    ▼
Reshape (128, 64)
    │
    ▼
UpSampling1D (size=4) → (512, 64)
    │
    ▼
UpSampling1D (size=4) → (2048, 64)
    │
    ▼
Conv1DTranspose (64 filters, kernel=15, padding='same')
    │
    ▼
UpSampling1D (size=4) → (8192, 64)
    │
    ▼
Conv1DTranspose (1 filter, kernel=7, padding='same', activation='sigmoid')
    │
    ▼
Output (8192, 1)
```

### Parameter count estimate
- Conv1D 1: 7 × 1 × 64 + 64 = 512
- Conv1D 2: 15 × 64 × 64 + 64 = 61,504
- Dense (8192→1024): 8,192 × 1024 + 1024 = 8.4M
- Dense (1024→256×2): 1024 × 512 + 512 = 525K
- Decoder Dense (256→1024): 256 × 1024 + 1024 = 263K
- Decoder Dense (1024→8192): 1024 × 8192 + 8192 = 8.4M
- Conv1DTranspose 1: 15 × 64 × 64 + 64 = 61,504
- Conv1DTranspose 2: 7 × 64 × 1 + 1 = 449

Total: ~18M parameters

---

## 2025-12-01 ~16:40: Switched to fully-connected architecture

### Problem

Conv1D layers caused CUDA errors:
1. OOM errors even with small batch sizes (JAX/TF allocator issues)
2. Integer overflow in CUDA kernel: `work_element_count >= 0 (-1673527296 vs. 0)`

The 8192-length 1D convolutions were too large for stable CUDA computation.

### Solution

Switched to fully-connected architecture:

**Encoder:**
```
Input (8192, 1) → Flatten
    → Dense(1024) → BN → LeakyReLU
    → Dense(512) → BN → LeakyReLU
    → z_mean(256), z_log_var(256)
```

**Decoder:**
```
(256) latent
    → Dense(512) → BN → LeakyReLU
    → Dense(1024) → BN → LeakyReLU
    → Dense(8192, sigmoid)
    → Reshape(8192, 1)
```

### Initial results

Training with 100k samples, batch size 512:
- Epoch 1: BCE 0.131 → Epoch 11: BCE 0.0015
- Loss dropping rapidly, model learning well

### Notes

- Using TensorFlow backend (not JAX) with `TF_GPU_ALLOCATOR=cuda_malloc_async`
- Will revisit Conv1D approach later with smaller kernel sizes or different architecture

---

## 2025-12-01 ~17:45: Memory leak diagnosis

### Problem

Training on full dataset (~4.8M samples) causes memory to grow until system runs out of RAM (~200GB consumed, then silent crash).

### Investigation with test_memory_scale.py

Created diagnostic script to track memory at different scales:

| Scale | Batches | Memory after 1 epoch |
|-------|---------|---------------------|
| 100k  | 195     | 10.15 GB            |
| 500k  | 976     | 26.36 GB            |
| 1M    | 1953    | (still growing)     |

Memory grows approximately linearly with number of batches processed. This is NOT related to:
- Memory-mapped file access (stays at ~0.03 GB)
- VAE model creation (~1 GB)
- Batch loading (~0.03 GB per batch)

### Root cause

TensorFlow/Keras accumulates internal state during training:
1. Gradient history for optimizer (Adam stores momentum/velocity)
2. XLA compilation cache grows
3. Potential tensor reference leaks in graph mode

### Solutions investigated

1. **`run_eagerly=True`**: Works (memory stable) but 10x slower (58ms vs 4ms per step)
2. **Clear `vae.losses`**: No effect
3. **`tf.function` with fixed signature**: No effect
4. **Manual training loop**: Same leak - ~21MB per batch

### Root cause found

The leak occurs when converting numpy arrays to TensorFlow tensors inside the training loop:
```python
batch = data[row_start:row_end, ...].astype(np.float32)  # Creates new Python object
train_step(batch)  # TF graph holds reference to this object
```

When using `tf.data.Dataset.from_tensor_slices` with pre-loaded data, memory is **stable**:
- 500k samples preloaded: 36.76 GB initial
- After creating dataset: 52.51 GB
- During training: 68.62 GB (stable throughout all 900 batches!)

### Solution

Pre-load data into memory as a single numpy array, then create `tf.data.Dataset.from_tensor_slices`. For full dataset (~4.8M samples × 8192 features × 4 bytes = ~157GB), this fits in 512GB RAM.

### Verified fix

Tested with 1M samples over 3 epochs:
- After loading data: 31.60 GB
- After dataset creation: 63.13 GB
- After epoch 1: 96.28 GB
- After epoch 2: 96.29 GB
- After epoch 3: 96.29 GB

Memory is **stable** during training - no more leak!

Updated VAE.py to use:
- `load_data_to_memory()` - loads data into numpy array
- `create_dataset()` - creates `tf.data.Dataset.from_tensor_slices()`

Expected memory for full dataset:
- Load ~4.3M train samples: ~140 GB
- Load ~0.5M val samples: ~16 GB
- TF dataset overhead: ~2x array size
- Training overhead: ~30 GB
- Total: ~350 GB (fits in 512 GB)

**Actual observation**: ~90% of 512 GB (~460 GB) during full dataset training. Higher than estimated due to TensorFlow's internal buffering for `prefetch` and `shuffle`. Memory is stable (no growth) during training.

Note: Added `with tf.device('/CPU:0')` to `create_dataset()` to prevent TensorFlow from trying to copy the entire dataset to GPU memory.

---

## 2025-12-01 ~19:00: Switched to JAX backend

Changed from TensorFlow to JAX backend for potentially better performance.

### Changes

1. Set `KERAS_BACKEND = 'jax'` instead of `'tensorflow'`
2. Replaced `tf.data.Dataset` with `NumpyBatchDataset(keras.utils.PyDataset)` - a simple indexed batch loader that works with any backend
3. Removed TensorFlow-specific environment variables and device placement code

### Data loading approach

Using `keras.utils.PyDataset` subclass that:
- Holds pre-loaded numpy data in memory
- Returns batches via `__getitem__(idx)`
- Shuffles indices on `on_epoch_end()` if shuffle=True
- No TensorFlow dependency

### Memory usage

JAX backend uses only ~30% of RAM (~154 GB) for the full dataset, compared to ~90% (~460 GB) with TensorFlow. Much more efficient.

---

## 2025-12-01 ~19:55: Convolutional VAE working

### Architecture implemented

Successfully implemented the Conv1D VAE architecture as specified:

**Encoder:**
```
Input (8192, 1)
    → Conv1D(64, kernel=7) → BN → LeakyReLU → MaxPool(4) → (2048, 64)
    → Conv1D(64, kernel=15) → BN → LeakyReLU → MaxPool(4) → (512, 64)
    → MaxPool(4) → (128, 64)
    → Flatten → 8192
    → Dense(1024) → BN → LeakyReLU
    → z_mean(256), z_log_var(256)
```

**Decoder:**
```
(256) latent
    → Dense(1024) → BN → LeakyReLU
    → Dense(8192) → BN → LeakyReLU → Reshape(128, 64)
    → UpSample(4) → (512, 64)
    → Conv1DTranspose(64, kernel=15) → BN → LeakyReLU → UpSample(4) → (2048, 64)
    → Conv1DTranspose(1, kernel=7, sigmoid) → UpSample(4) → (8192, 1)
```

### Parameters
- Encoder: ~9M parameters
- Decoder: ~8.7M parameters
- Total: ~17.7M parameters

### Test results (50k samples, 10 epochs)

| Epoch | BCE | Val Loss | KL |
|-------|-----|----------|-----|
| 1 | 0.371 | 3038 | 39.8 |
| 5 | 0.009 | 82.9 | 74.7 |
| 10 | 0.003 | 29.7 | 49.5 |

BCE reduced by ~100x over 10 epochs. Model is learning well.

### Fixes applied

1. **VAEMetricsCallback OOM**: The callback was processing 5000 validation samples at once, causing GPU OOM. Fixed by processing encoder and decoder in batches of 256.

### Notes

- Using JAX backend
- Training speed: ~1 minute per 10 epochs for 50k samples
- Ready for full dataset training

---

## 2025-12-01 ~21:40: Posterior collapse observed

### Problem

During full dataset training, KL divergence collapsed from ~11 to <1 after epoch 5:

| Epoch | KL | w = KL × 0.1 |
|-------|-----|--------------|
| 6 | 1.86 | 0.19 |
| 8 | 1.05 | 0.11 |
| 10 | 0.58 | 0.06 |
| 11 | 0.77 | 0.08 |

This indicates **posterior collapse** - the encoder outputs near-zero variance, making all latent codes nearly identical. The decoder is powerful enough to reconstruct without using the latent space meaningfully.

### Fix: Reduced decoder capacity

Changed conv layers from 64 filters (kernel 7/15) to 32 filters (kernel 5/5):

**Encoder:**
```
Input (8192, 1)
    → Conv1D(32, kernel=5) → MaxPool(4) → (2048, 32)
    → Conv1D(32, kernel=5) → MaxPool(4) → (512, 32)
    → MaxPool(4) → (128, 32)
    → Flatten → 4096
    → Dense(1024) → z_mean(256), z_log_var(256)
```

**Decoder:**
```
(256) latent
    → Dense(1024) → Dense(4096) → Reshape(128, 32)
    → UpSample(4) → (512, 32)
    → Conv1DTranspose(32, kernel=5) → UpSample(4) → (2048, 32)
    → Conv1DTranspose(1, kernel=5, sigmoid) → UpSample(4) → (8192, 1)
```

This reduces parameter count and forces the model to rely more on the latent space for reconstruction.

---

## 2025-12-02 ~09:58: First full training completed (with posterior collapse)

The 200-epoch training completed with the 64-filter architecture:

- **Final KL**: 0.56 (very low - indicates posterior collapse)
- **Final BCE**: 0.001202
- **Final Val loss**: 9.92

The model reconstructs well but the latent space is likely not useful for clustering due to posterior collapse.

---

## 2025-12-02 ~10:00: Further architecture reduction

Per Torben's request, reduced architecture further:

**Encoder:**
```
Input (8192, 1)
    → Conv1D(16, kernel=3) → MaxPool(4) → (2048, 16)
    → Conv1D(32, kernel=5) → MaxPool(16) → (128, 32)
    → Flatten → 4096
    → Dense(1024) → z_mean(256), z_log_var(256)
```

**Decoder:**
```
(256) latent
    → Dense(1024) → Dense(4096) → Reshape(128, 32)
    → UpSample(16) → (2048, 32)
    → Conv1DTranspose(16, kernel=5) → UpSample(4) → (8192, 16)
    → Conv1DTranspose(1, kernel=3, sigmoid) → (8192, 1)
```

### Changes from previous
- First conv: 32 filters → 16 filters, kernel 5 → 3
- Combined two MaxPool(4) into one MaxPool(16)
- Mirrored changes in decoder

### Parameters
- Encoder: 4.7M
- Decoder: 4.5M
- Total: ~9.2M (same as before - dense layers dominate)