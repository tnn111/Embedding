#!/usr/bin/env python3
"""
Log-Space Variational Autoencoder for multi-scale k-mer frequency distributions.

Handles multi-dimensional inputs (INPUT_DIM features): sequence length, 7-mer, 4-mer,
3-mer frequencies, and GC content. Optimized for Keras 3 / JAX with 2M+ sequences.
"""

import argparse
import os
os.environ['KERAS_BACKEND'] = 'jax'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import  logging
import  numpy                   as      np
import  keras
from    keras                   import  layers
from    keras                   import  Model
from    keras                   import  ops
from    sklearn.model_selection import train_test_split
import  pickle

# Setup logging to both stdout and file
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(message)s',
    datefmt = '%Y-%m-%d %H:%M:%S',
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler('vae_training.log', mode = 'a')
    ]
)
logger = logging.getLogger(__name__)

# Global constants
INPUT_DIM = 8362   # sequence length + 7-mer + 4-mer + 3-mer frequencies + GC content
OUTPUT_DIM = 8361  # decoder output (no length): 7-mers + 4-mers + 3-mers + GC
LATENT_DIM = 256
SEED = 42

# Feature slice indices (for input tensor)
# Input layout: [length(1), 7-mers(8192), 4-mers(136), 3-mers(32), GC(1)]
LENGTH_SLICE = (0, 1)           # index 0
KMER_7_SLICE = (1, 8193)        # indices 1-8192 (8,192 features)
KMER_4_SLICE = (8193, 8329)     # indices 8193-8328 (136 features)
KMER_3_SLICE = (8329, 8361)     # indices 8329-8360 (32 features)
GC_SLICE = (8361, 8362)         # index 8361

# Output slice indices (no length field)
# Output layout: [7-mers(8192), 4-mers(136), 3-mers(32), GC(1)]
OUT_KMER_7_SLICE = (0, 8192)      # 8,192 features
OUT_KMER_4_SLICE = (8192, 8328)   # 136 features
OUT_KMER_3_SLICE = (8328, 8360)   # 32 features
OUT_GC_SLICE = (8360, 8361)       # 1 feature


class VAEMetricsCallback(keras.callbacks.Callback):
    """Track VAE metrics outside of JIT-compiled code (Keras 3 compatible)."""

    def __init__(self, validation_data: tuple[np.ndarray, np.ndarray], sample_size: int = 5000):
        super().__init__()
        self.validation_data = validation_data
        self.sample_size = sample_size
        self._sample_idx: np.ndarray | None = None

    def on_train_begin(self, logs = None):
        """Create fixed sample indices for consistent metrics across epochs."""
        if self.validation_data is not None:
            n_samples = min(self.sample_size, len(self.validation_data[0]))
            self._sample_idx = np.random.choice(len(self.validation_data[0]), n_samples, replace = False)

    def on_epoch_end(self, epoch, logs = None):
        if logs is None or self.validation_data is None or self._sample_idx is None:
            return

        sample_x = self.validation_data[0][self._sample_idx]

        z_mean, z_log_var, z = self.model.encoder(sample_x, training = False)
        kl_loss = -0.5 * float(ops.mean(ops.sum(1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var), axis = 1)))  # type: ignore[arg-type]
        weighted_kl = float(self.model.kl_weight) * kl_loss

        # Get decoder outputs including gate and values
        reconstruction, gate_7, values_7 = self.model.decoder(z, training = False)
        target = sample_x[:, 1:]  # Skip length field

        # 7-mer metrics with gated sparsity
        target_7 = target[:, OUT_KMER_7_SLICE[0]:OUT_KMER_7_SLICE[1]]
        pred_7 = reconstruction[:, OUT_KMER_7_SLICE[0]:OUT_KMER_7_SLICE[1]]
        sq_err_7 = ops.square(target_7 - pred_7)

        # Sparsity classification metrics
        target_nonzero = ops.cast(target_7 > -10.0, 'float32')
        zero_mask = 1.0 - target_nonzero
        nonzero_count = ops.maximum(ops.sum(target_nonzero), 1.0)
        zero_count = ops.maximum(ops.sum(zero_mask), 1.0)

        # Gate accuracy: how well does gate predict zero vs non-zero?
        gate_pred_nonzero = ops.cast(gate_7 > 0.5, 'float32')
        gate_accuracy = float(ops.mean(ops.cast(gate_pred_nonzero == target_nonzero, 'float32')))  # type: ignore[arg-type]

        # BCE for sparsity
        gate_clipped = ops.clip(gate_7, 1e-7, 1.0 - 1e-7)
        sparsity_bce = float(-ops.mean(
            target_nonzero * ops.log(gate_clipped) +
            (1.0 - target_nonzero) * ops.log(1.0 - gate_clipped)
        ))  # type: ignore[arg-type]

        # MSE on values (for non-zero targets only)
        mse_7_values = float(ops.sum(target_nonzero * ops.square(target_7 - values_7)) / nonzero_count)  # type: ignore[arg-type]

        # Overall 7-mer MSE (on gated output)
        mse_7_nonzero = float(ops.sum(sq_err_7 * target_nonzero) / nonzero_count)  # type: ignore[arg-type]
        mse_7_zero = float(ops.sum(sq_err_7 * zero_mask) / zero_count)  # type: ignore[arg-type]
        mse_7 = float(ops.mean(sq_err_7))  # type: ignore[arg-type]
        pct_nonzero = float(ops.sum(target_nonzero) / (ops.shape(target_7)[0] * ops.shape(target_7)[1]) * 100)  # type: ignore[arg-type]

        mse_4 = float(ops.mean(ops.square(target[:, OUT_KMER_4_SLICE[0]:OUT_KMER_4_SLICE[1]] -
                                          reconstruction[:, OUT_KMER_4_SLICE[0]:OUT_KMER_4_SLICE[1]])))  # type: ignore[arg-type]
        mse_3 = float(ops.mean(ops.square(target[:, OUT_KMER_3_SLICE[0]:OUT_KMER_3_SLICE[1]] -
                                          reconstruction[:, OUT_KMER_3_SLICE[0]:OUT_KMER_3_SLICE[1]])))  # type: ignore[arg-type]
        mse_gc = float(ops.mean(ops.square(target[:, OUT_GC_SLICE[0]:OUT_GC_SLICE[1]] -
                                           reconstruction[:, OUT_GC_SLICE[0]:OUT_GC_SLICE[1]])))  # type: ignore[arg-type]

        # Total recon loss (using new gated loss formulation)
        loss_7 = sparsity_bce + mse_7_values
        recon_loss = (loss_7 + mse_4 + mse_3 + mse_gc) * (OUTPUT_DIM / 4)

        val_loss = logs.get('val_loss')
        val_loss_str = f'{val_loss:.2f}' if val_loss is not None else 'N/A'
        logger.info(
            f'Epoch {epoch + 1}/{self.params["epochs"]}: Recon: {recon_loss:.2f}, KL: {kl_loss:.2f} (w={weighted_kl:.2f}), Val: {val_loss_str} | '
            f'7mer: BCE={sparsity_bce:.4f}, valMSE={mse_7_values:.4f}, gateAcc={gate_accuracy:.1%} | '
            f'outMSE={mse_7:.4f} (nz={mse_7_nonzero:.4f}, z={mse_7_zero:.4f}, {pct_nonzero:.1f}%nz), 4mer={mse_4:.4f}, 3mer={mse_3:.4f}, GC={mse_gc:.4f}'
        )


class KLWarmupCallback(keras.callbacks.Callback):
    def __init__(self, warmup_epochs = 10, max_weight = 1.0, skip_warmup = False):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.max_weight = max_weight
        self.skip_warmup = skip_warmup

    def on_epoch_begin(self, epoch, logs = None):
        if self.skip_warmup or epoch >= self.warmup_epochs:
            new_weight = self.max_weight
        else:
            new_weight = (epoch / self.warmup_epochs) * self.max_weight
        self.model.kl_weight.assign(new_weight)


class VAECheckpoint(keras.callbacks.Callback):
    """Save VAE, encoder, and decoder models when validation loss improves."""

    def __init__(self, filepath_prefix = 'vae', monitor = 'val_loss', verbose = 0, initial_best = None):
        super().__init__()
        self.filepath_prefix = filepath_prefix
        self.monitor = monitor
        self.verbose = verbose
        self.best = initial_best if initial_best is not None else float('inf')

    def on_epoch_end(self, epoch, logs = None):
        if logs is None:
            return

        current = logs.get(self.monitor)
        if current is None:
            return

        if current < self.best:
            self.best = current
            self.model.save(f'{self.filepath_prefix}_best.keras')
            self.model.encoder.save(f'{self.filepath_prefix}_encoder_best.keras')
            self.model.decoder.save(f'{self.filepath_prefix}_decoder_best.keras')
            if self.verbose:
                logger.info(f'Epoch {epoch + 1}: {self.monitor} improved to {current:.4f}, saved models')


class Sampling(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(SEED)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape = (batch, dim), seed = self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon

    def get_config(self):
        return super().get_config()


class ClipLayer(layers.Layer):
    """Clips tensor values to a specified range."""

    def __init__(self, min_value = -20, max_value = 2, **kwargs):
        super().__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value

    def call(self, inputs):
        return ops.clip(inputs, self.min_value, self.max_value)

    def get_config(self):
        config = super().get_config()
        config.update({'min_value': self.min_value, 'max_value': self.max_value})
        return config


class SliceLayer(layers.Layer):
    """Slices a tensor along the last axis."""

    def __init__(self, start, end, **kwargs):
        super().__init__(**kwargs)
        self.start = start
        self.end = end

    def call(self, inputs):
        return inputs[:, self.start:self.end]

    def get_config(self):
        config = super().get_config()
        config.update({'start': self.start, 'end': self.end})
        return config


class GatedCombineLayer(layers.Layer):
    """Combines gate and values: gate * values + (1 - gate) * floor_value."""

    def __init__(self, floor_value = -13.8, **kwargs):
        super().__init__(**kwargs)
        self.floor_value = floor_value

    def call(self, inputs):
        gate, values = inputs
        return gate * values + (1.0 - gate) * self.floor_value

    def get_config(self):
        config = super().get_config()
        config.update({'floor_value': self.floor_value})
        return config


class VAE(Model):
    def __init__(self, latent_dim = LATENT_DIM, kl_weight = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.kl_weight = keras.Variable(kl_weight, trainable = False, dtype = 'float32', name = 'kl_weight')
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

    def _build_encoder(self):
        """Build multi-branch encoder for 7-mer, 4-mer, and 3-mer features.

        Architecture:
            - 7-mer branch: 8,194 → 512 → 256 (73% of concat)
            - 4-mer branch: 138 → 128 → 64 (18% of concat)
            - 3-mer branch: 34 → 64 → 32 (9% of concat)
            - Concatenate (352) → 512 → latent (256)
        """
        encoder_inputs = keras.Input(shape = (INPUT_DIM,))

        # Extract features from input
        length = SliceLayer(*LENGTH_SLICE, name = 'slice_length')(encoder_inputs)
        kmers_7 = SliceLayer(*KMER_7_SLICE, name = 'slice_7mer')(encoder_inputs)
        kmers_4 = SliceLayer(*KMER_4_SLICE, name = 'slice_4mer')(encoder_inputs)
        kmers_3 = SliceLayer(*KMER_3_SLICE, name = 'slice_3mer')(encoder_inputs)
        gc = SliceLayer(*GC_SLICE, name = 'slice_gc')(encoder_inputs)

        # Build branch inputs: k-mers + length + GC
        b_7 = layers.Concatenate(name = 'branch_7_input')([kmers_7, length, gc])  # (batch, 8194)
        b_4 = layers.Concatenate(name = 'branch_4_input')([kmers_4, length, gc])  # (batch, 138)
        b_3 = layers.Concatenate(name = 'branch_3_input')([kmers_3, length, gc])  # (batch, 34)

        # 7-mer branch: 8,194 → 512 → 256
        x_7 = layers.Dense(512, name = 'enc_7mer_dense1')(b_7)
        x_7 = layers.BatchNormalization(name = 'enc_7mer_bn1')(x_7)
        x_7 = layers.LeakyReLU(negative_slope = 0.2, name = 'enc_7mer_relu1')(x_7)
        x_7 = layers.Dense(256, name = 'enc_7mer_dense2')(x_7)
        x_7 = layers.BatchNormalization(name = 'enc_7mer_bn2')(x_7)
        x_7 = layers.LeakyReLU(negative_slope = 0.2, name = 'enc_7mer_relu2')(x_7)

        # 4-mer branch: 138 → 128 → 64
        x_4 = layers.Dense(128, name = 'enc_4mer_dense1')(b_4)
        x_4 = layers.BatchNormalization(name = 'enc_4mer_bn1')(x_4)
        x_4 = layers.LeakyReLU(negative_slope = 0.2, name = 'enc_4mer_relu1')(x_4)
        x_4 = layers.Dense(64, name = 'enc_4mer_dense2')(x_4)
        x_4 = layers.BatchNormalization(name = 'enc_4mer_bn2')(x_4)
        x_4 = layers.LeakyReLU(negative_slope = 0.2, name = 'enc_4mer_relu2')(x_4)

        # 3-mer branch: 34 → 64 → 32
        x_3 = layers.Dense(64, name = 'enc_3mer_dense1')(b_3)
        x_3 = layers.BatchNormalization(name = 'enc_3mer_bn1')(x_3)
        x_3 = layers.LeakyReLU(negative_slope = 0.2, name = 'enc_3mer_relu1')(x_3)
        x_3 = layers.Dense(32, name = 'enc_3mer_dense2')(x_3)
        x_3 = layers.BatchNormalization(name = 'enc_3mer_bn2')(x_3)
        x_3 = layers.LeakyReLU(negative_slope = 0.2, name = 'enc_3mer_relu2')(x_3)

        # Concatenate branches: 256 + 64 + 32 = 352
        x = layers.Concatenate(name = 'enc_concat')([x_7, x_4, x_3])

        # Shared layers: 352 → 512
        x = layers.Dense(512, name = 'enc_shared_dense')(x)
        x = layers.BatchNormalization(name = 'enc_shared_bn')(x)
        x = layers.LeakyReLU(negative_slope = 0.2, name = 'enc_shared_relu')(x)

        # Latent space
        z_mean = layers.Dense(self.latent_dim, name = 'z_mean')(x)
        z_log_var = layers.Dense(self.latent_dim, name = 'z_log_var')(x)
        z_log_var = ClipLayer(name = 'z_log_var_clip')(z_log_var)
        z = Sampling()([z_mean, z_log_var])

        return Model(encoder_inputs, [z_mean, z_log_var, z], name = 'encoder')

    def _build_decoder(self):
        """Build multi-branch decoder with gated sparsity for 7-mers.

        Architecture:
            - Latent (256) → 512 → 352 (shared)
            - Split to branches via separate Dense projections
            - 7-mer branch: 256 → 512 → values(8192) + gate(8192), gated output
            - 4-mer branch: 64 → 128 → 136 k-mers + 1 GC
            - 3-mer branch: 32 → 64 → 32 k-mers + 1 GC
            - Average 3 GC predictions
            - Output: ([7-mers, 4-mers, 3-mers, GC], gate_7) where gate_7 is sigmoid

        The 7-mer gated output = gate * values + (1 - gate) * FLOOR_VALUE
        This separates sparsity prediction (gate) from value prediction.
        """
        latent_inputs = keras.Input(shape = (self.latent_dim,))

        # Shared layers: 256 → 512 → 352
        x = layers.Dense(512, name = 'dec_shared_dense1')(latent_inputs)
        x = layers.BatchNormalization(name = 'dec_shared_bn1')(x)
        x = layers.LeakyReLU(negative_slope = 0.2, name = 'dec_shared_relu1')(x)
        x = layers.Dense(352, name = 'dec_shared_dense2')(x)
        x = layers.BatchNormalization(name = 'dec_shared_bn2')(x)
        x = layers.LeakyReLU(negative_slope = 0.2, name = 'dec_shared_relu2')(x)

        # 7-mer branch with gated sparsity: 256 → 512 → values + gate
        x_7 = layers.Dense(256, name = 'dec_7mer_dense1')(x)
        x_7 = layers.BatchNormalization(name = 'dec_7mer_bn1')(x_7)
        x_7 = layers.LeakyReLU(negative_slope = 0.2, name = 'dec_7mer_relu1')(x_7)
        x_7 = layers.Dense(512, name = 'dec_7mer_dense2')(x_7)
        x_7 = layers.BatchNormalization(name = 'dec_7mer_bn2')(x_7)
        x_7 = layers.LeakyReLU(negative_slope = 0.2, name = 'dec_7mer_relu2')(x_7)

        # Gated sparsity: separate value and gate predictions
        values_7 = layers.Dense(8192, activation = 'linear', name = 'dec_7mer_values')(x_7)
        gate_logits_7 = layers.Dense(8192, name = 'dec_7mer_gate_logits')(x_7)
        gate_7 = layers.Activation('sigmoid', name = 'dec_7mer_gate')(gate_logits_7)

        # Gated output: gate * values + (1 - gate) * floor
        # FLOOR_VALUE = log(1e-6) ≈ -13.8
        kmers_7 = GatedCombineLayer(floor_value = -13.8, name = 'dec_7mer_gated_output')([gate_7, values_7])

        gc_7 = layers.Dense(1, activation = 'linear', name = 'dec_gc_7')(x_7)

        # 4-mer branch: 64 → 128 → outputs
        x_4 = layers.Dense(64, name = 'dec_4mer_dense1')(x)
        x_4 = layers.BatchNormalization(name = 'dec_4mer_bn1')(x_4)
        x_4 = layers.LeakyReLU(negative_slope = 0.2, name = 'dec_4mer_relu1')(x_4)
        x_4 = layers.Dense(128, name = 'dec_4mer_dense2')(x_4)
        x_4 = layers.BatchNormalization(name = 'dec_4mer_bn2')(x_4)
        x_4 = layers.LeakyReLU(negative_slope = 0.2, name = 'dec_4mer_relu2')(x_4)
        kmers_4 = layers.Dense(136, activation = 'linear', name = 'dec_4mer_out')(x_4)
        gc_4 = layers.Dense(1, activation = 'linear', name = 'dec_gc_4')(x_4)

        # 3-mer branch: 32 → 64 → outputs
        x_3 = layers.Dense(32, name = 'dec_3mer_dense1')(x)
        x_3 = layers.BatchNormalization(name = 'dec_3mer_bn1')(x_3)
        x_3 = layers.LeakyReLU(negative_slope = 0.2, name = 'dec_3mer_relu1')(x_3)
        x_3 = layers.Dense(64, name = 'dec_3mer_dense2')(x_3)
        x_3 = layers.BatchNormalization(name = 'dec_3mer_bn2')(x_3)
        x_3 = layers.LeakyReLU(negative_slope = 0.2, name = 'dec_3mer_relu2')(x_3)
        kmers_3 = layers.Dense(32, activation = 'linear', name = 'dec_3mer_out')(x_3)
        gc_3 = layers.Dense(1, activation = 'linear', name = 'dec_gc_3')(x_3)

        # Average GC predictions from all branches
        gc_avg = layers.Average(name = 'dec_gc_avg')([gc_7, gc_4, gc_3])

        # Concatenate outputs: [7-mers(8192), 4-mers(136), 3-mers(32), GC(1)] = 8,361
        decoder_outputs = layers.Concatenate(name = 'dec_output')([kmers_7, kmers_4, kmers_3, gc_avg])

        # Return both reconstruction and gate for loss computation
        return Model(latent_inputs, [decoder_outputs, gate_7, values_7], name = 'decoder')

    def call(self, inputs, training = None):
        z_mean, z_log_var, z = self.encoder(inputs, training = training)
        reconstruction, gate_7, values_7 = self.decoder(z, training = training)

        # Target is input without length field: [7-mers, 4-mers, 3-mers, GC] = 8,361 features
        target = inputs[:, 1:]  # Skip index 0 (length)

        # 7-mer gated sparsity loss: BCE for gate + MSE for values (on non-zeros only)
        target_7 = target[:, OUT_KMER_7_SLICE[0]:OUT_KMER_7_SLICE[1]]
        target_nonzero = ops.cast(target_7 > -10.0, 'float32')  # 1 if non-zero, 0 if zero

        # BCE loss for sparsity prediction (which 7-mers are present?)
        # Clip gate to avoid log(0)
        gate_clipped = ops.clip(gate_7, 1e-7, 1.0 - 1e-7)
        sparsity_bce = -ops.mean(
            target_nonzero * ops.log(gate_clipped) +
            (1.0 - target_nonzero) * ops.log(1.0 - gate_clipped)
        )

        # MSE loss for value prediction (only on non-zero targets)
        nonzero_count = ops.maximum(ops.sum(target_nonzero), 1.0)
        mse_7_values = ops.sum(target_nonzero * ops.square(target_7 - values_7)) / nonzero_count

        # Combined 7-mer loss (BCE and MSE have similar scales, ~0.5-1.0 each)
        loss_7 = sparsity_bce + mse_7_values

        # 4-mer, 3-mer, GC: standard MSE
        mse_4 = ops.mean(ops.square(target[:, OUT_KMER_4_SLICE[0]:OUT_KMER_4_SLICE[1]] -
                                    reconstruction[:, OUT_KMER_4_SLICE[0]:OUT_KMER_4_SLICE[1]]))
        mse_3 = ops.mean(ops.square(target[:, OUT_KMER_3_SLICE[0]:OUT_KMER_3_SLICE[1]] -
                                    reconstruction[:, OUT_KMER_3_SLICE[0]:OUT_KMER_3_SLICE[1]]))
        mse_gc = ops.mean(ops.square(target[:, OUT_GC_SLICE[0]:OUT_GC_SLICE[1]] -
                                     reconstruction[:, OUT_GC_SLICE[0]:OUT_GC_SLICE[1]]))

        # Scale reconstruction loss to similar magnitude as before
        recon_loss = (loss_7 + mse_4 + mse_3 + mse_gc) * (OUTPUT_DIM / 4)

        kl_loss = -0.5 * ops.mean(ops.sum(1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var), axis = 1))

        total_loss = recon_loss + (self.kl_weight * kl_loss)
        self.add_loss(total_loss)

        return reconstruction

    def get_config(self):
        config = super().get_config()
        config.update({
            'latent_dim': self.latent_dim,
            'kl_weight': float(self.kl_weight.numpy()),
        })
        return config


def load_data_log_space(file_path, expected_dim = INPUT_DIM):
    """Load k-mer frequency data and transform to log-space.

    Args:
        file_path: Path to the .npy file containing frequency data.
        expected_dim: Expected number of features per sample.

    Returns:
        Log-transformed data as float32 array.

    Raises:
        ValueError: If the data dimensions don't match expected_dim.
    """
    logger.info(f'Loading data from {file_path}...')

    # Memory-map the file to avoid loading entire array into RAM at once
    data = np.load(file_path, mmap_mode = 'r')

    # Validate input dimensions
    if data.ndim != 2:
        raise ValueError(f'Expected 2D array, got {data.ndim}D array with shape {data.shape}')
    if data.shape[1] != expected_dim:
        raise ValueError(f'Expected {expected_dim} features, got {data.shape[1]}. '
                         f'Data shape: {data.shape}')

    logger.info(f'Found {len(data)} samples, transforming to Log-Space in chunks...')

    # Pre-allocate output array
    data_log = np.empty(data.shape, dtype = np.float32)

    # Process in chunks to reduce peak memory usage
    chunk_size = 100_000
    for start in range(0, len(data), chunk_size):
        end = min(start + chunk_size, len(data))
        data_log[start:end] = np.log(data[start:end].astype(np.float32) + 1e-6)
        if (start // chunk_size) % 5 == 0:
            logger.info(f'  Processed {end:,} / {len(data):,} samples')

    logger.info(f'Loaded {len(data_log):,} samples')
    logger.info(f'Data stats: Min {data_log.min():.2f}, Max {data_log.max():.2f}, Mean {data_log.mean():.2f}')
    return data_log


def main():
    parser = argparse.ArgumentParser(description = 'Train a VAE on k-mer frequency data')
    parser.add_argument('-i', '--input', required = True, help = 'Path to input .npy file with k-mer frequencies')
    parser.add_argument('-e', '--epochs', type = int, default = 100, help = 'Number of training epochs (default: 100)')
    parser.add_argument('-l', '--learning-rate', type = float, default = 1e-4, help = 'Learning rate (default: 1e-4)')
    parser.add_argument('-b', '--batch-size', type = int, default = 4096, help = 'Batch size (default: 4096)')
    args = parser.parse_args()

    np.random.seed(SEED)
    keras.utils.set_random_seed(SEED)

    # Load Log-Transformed Data
    X = load_data_log_space(args.input)

    # Split
    X_train, X_val = train_test_split(X, test_size = 0.2, random_state = SEED)
    logger.info(f'Training on {X_train.shape[0]} samples, validating on {X_val.shape[0]} samples')

    # Load existing model if available, otherwise create new one
    resuming = os.path.exists('vae_best.keras')
    initial_best = None

    if resuming:
        logger.info('Loading existing model from vae_best.keras...')
        vae = keras.models.load_model('vae_best.keras', custom_objects = {
            'VAE': VAE, 'Sampling': Sampling, 'ClipLayer': ClipLayer,
            'SliceLayer': SliceLayer, 'GatedCombineLayer': GatedCombineLayer
        })
        logger.info('Model loaded successfully. Recompiling and continuing training...')
        # Load previous best val_loss if history exists
        if os.path.exists('vae_history.pkl'):
            with open('vae_history.pkl', 'rb') as f:
                prev_history = pickle.load(f)
                if 'val_loss' in prev_history and prev_history['val_loss']:
                    initial_best = min(prev_history['val_loss'])
                    logger.info(f'Previous best val_loss: {initial_best:.4f}')
    else:
        logger.info('No existing model found. Creating new VAE...')
        vae = VAE(latent_dim = LATENT_DIM)

    # Always (re)compile to ensure consistent optimizer settings
    vae.compile(optimizer = keras.optimizers.Adam(learning_rate = args.learning_rate))  # type: ignore[arg-type]

    # Setup callbacks
    vae_metrics = VAEMetricsCallback(validation_data = (X_val, X_val), sample_size = 5000)

    callbacks = [
        KLWarmupCallback(warmup_epochs = 5, max_weight = 1.0, skip_warmup = resuming),
        vae_metrics,
        VAECheckpoint(filepath_prefix = 'vae', monitor = 'val_loss', verbose = 0, initial_best = initial_best),
        keras.callbacks.ReduceLROnPlateau(
            monitor = 'val_loss',
            factor = 0.5,
            patience = 20,
            min_lr = 1e-6,
            verbose = 1
        )
    ]

    logger.info('Starting Training...')
    history = vae.fit(
        X_train,
        X_train,  # Input = Output (Log Space)
        epochs = args.epochs,
        batch_size = args.batch_size,
        validation_data = (X_val, X_val),
        callbacks = callbacks,
        verbose = 0  # type: ignore[arg-type]
    )

    # Save final models
    vae.save('vae_final.keras')
    vae.encoder.save('vae_encoder_final.keras')
    vae.decoder.save('vae_decoder_final.keras')
    with open('vae_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    logger.info(f'Training complete. Final val_loss: {history.history["val_loss"][-1]:.4f}')
    logger.info('Models saved: vae_final.keras, vae_encoder_final.keras, vae_decoder_final.keras')

if __name__ == '__main__':
    main()
