#!/usr/bin/env python3
"""
Variational Autoencoder for multi-scale k-mer frequency distributions using BCE loss.

K-mer frequencies are normalized distributions (sum to 1 within each group), making them
suitable for BCE loss which treats each frequency as a probability.

Input format (from convert_txt_to_npy):
    length(1) + 6-mers(2080) + 5-mers(512) + 4-mers(136) + 3-mers(32) + GC(1) = 2,762

Length column is skipped during loading - only k-mer frequencies and GC are used.
"""

import argparse
import os
os.environ['KERAS_BACKEND'] = 'jax'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import logging
import numpy as np
import keras
from keras import layers
from keras import Model
from keras import ops
from sklearn.model_selection import train_test_split
import pickle

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
# Input/Output: 6-mers(2080) + 5-mers(512) + 4-mers(136) + 3-mers(32) + GC(1) = 2,761
# (length column from data file is skipped during loading)
INPUT_DIM = 2761
OUTPUT_DIM = 2761
LATENT_DIM = 256
SEED = 42

# Feature slice indices
# Layout: [6-mers(2080), 5-mers(512), 4-mers(136), 3-mers(32), GC(1)]
KMER_6_SLICE = (0, 2080)        # 2,080 features
KMER_5_SLICE = (2080, 2592)     # 512 features
KMER_4_SLICE = (2592, 2728)     # 136 features
KMER_3_SLICE = (2728, 2760)     # 32 features
GC_SLICE = (2760, 2761)         # 1 feature


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
        kl_loss = -0.5 * float(ops.mean(ops.sum(1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var), axis = 1)))
        weighted_kl = float(self.model.kl_weight) * kl_loss

        reconstruction = self.model.decoder(z, training = False)
        target = sample_x  # Input and output are now the same (no length)

        # Per-group metrics
        eps = 1e-7
        pred_clipped = ops.clip(reconstruction, eps, 1.0 - eps)

        def bce(t, p):
            return -ops.mean(t * ops.log(p) + (1.0 - t) * ops.log(1.0 - p))

        # BCE for 6-mers, 5-mers, 4-mers
        bce_6 = float(bce(target[:, KMER_6_SLICE[0]:KMER_6_SLICE[1]],
                         pred_clipped[:, KMER_6_SLICE[0]:KMER_6_SLICE[1]]))
        # MSE in log-space for 5-mers (add 0.01 offset)
        target_5 = target[:, KMER_5_SLICE[0]:KMER_5_SLICE[1]]
        pred_5 = reconstruction[:, KMER_5_SLICE[0]:KMER_5_SLICE[1]]
        mse_5 = float(ops.mean(ops.square(ops.log(target_5 + 0.01) - ops.log(pred_5 + 0.01))))

        # MSE in log-space for 4-mers (add 0.01 offset)
        target_4 = target[:, KMER_4_SLICE[0]:KMER_4_SLICE[1]]
        pred_4 = reconstruction[:, KMER_4_SLICE[0]:KMER_4_SLICE[1]]
        mse_4 = float(ops.mean(ops.square(ops.log(target_4 + 0.01) - ops.log(pred_4 + 0.01))))

        # MSE in log-space for 3-mers (add 0.01 offset)
        target_3 = target[:, KMER_3_SLICE[0]:KMER_3_SLICE[1]]
        pred_3 = reconstruction[:, KMER_3_SLICE[0]:KMER_3_SLICE[1]]
        mse_3 = float(ops.mean(ops.square(ops.log(target_3 + 0.01) - ops.log(pred_3 + 0.01))))

        # MSE for GC using logit transform: log(x / (1-x))
        target_gc = ops.clip(target[:, GC_SLICE[0]:GC_SLICE[1]], eps, 1.0 - eps)
        pred_gc = ops.clip(reconstruction[:, GC_SLICE[0]:GC_SLICE[1]], eps, 1.0 - eps)
        mse_gc = float(ops.mean(ops.square(ops.log(target_gc / (1.0 - target_gc)) - ops.log(pred_gc / (1.0 - pred_gc)))))

        # Total reconstruction loss (approximate)
        recon_loss = bce_6 * OUTPUT_DIM * 100 + (mse_5 + mse_4 + mse_3 + mse_gc) * OUTPUT_DIM * 100 / 4

        val_loss = logs.get('val_loss')
        val_loss_str = f'{val_loss:.2f}' if val_loss is not None else 'N/A'
        logger.info(
            f'Epoch {epoch + 1}/{self.params["epochs"]}: Recon: {recon_loss:.2f}, KL: {kl_loss:.2f} (w={weighted_kl:.2f}), Val: {val_loss_str} | '
            f'6mer={bce_6:.4f}, 5mer={mse_5:.4f}, 4mer={mse_4:.4f}, 3mer={mse_3:.4f}, GC={mse_gc:.4f}'
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


class VAE(Model):
    def __init__(self, latent_dim = LATENT_DIM, kl_weight = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.kl_weight = keras.Variable(kl_weight, trainable = False, dtype = 'float32', name = 'kl_weight')
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

    def _build_encoder(self):
        """Build simple encoder: input → hidden → latent.

        Length is included as advisory input but not reconstructed.

        Architecture:
            - Input (2,762) → 1024 → 512 → latent (256)
        """
        encoder_inputs = keras.Input(shape = (INPUT_DIM,))

        x = layers.Dense(1024, name = 'enc_dense1')(encoder_inputs)
        x = layers.BatchNormalization(name = 'enc_bn1')(x)
        x = layers.LeakyReLU(negative_slope = 0.2, name = 'enc_relu1')(x)

        x = layers.Dense(512, name = 'enc_dense2')(x)
        x = layers.BatchNormalization(name = 'enc_bn2')(x)
        x = layers.LeakyReLU(negative_slope = 0.2, name = 'enc_relu2')(x)

        # Latent space
        z_mean = layers.Dense(self.latent_dim, name = 'z_mean')(x)
        z_log_var = layers.Dense(self.latent_dim, name = 'z_log_var')(x)
        z_log_var = ClipLayer(name = 'z_log_var_clip')(z_log_var)
        z = Sampling()([z_mean, z_log_var])

        return Model(encoder_inputs, [z_mean, z_log_var, z], name = 'encoder')

    def _build_decoder(self):
        """Build simple decoder: latent → hidden → output with sigmoid.

        Architecture:
            - Latent (256) → 512 → 1024 → output (2,761) with sigmoid
        """
        latent_inputs = keras.Input(shape = (self.latent_dim,))

        x = layers.Dense(512, name = 'dec_dense1')(latent_inputs)
        x = layers.BatchNormalization(name = 'dec_bn1')(x)
        x = layers.LeakyReLU(negative_slope = 0.2, name = 'dec_relu1')(x)

        x = layers.Dense(1024, name = 'dec_dense2')(x)
        x = layers.BatchNormalization(name = 'dec_bn2')(x)
        x = layers.LeakyReLU(negative_slope = 0.2, name = 'dec_relu2')(x)

        # Output layer: sigmoid activation for BCE loss
        decoder_outputs = layers.Dense(OUTPUT_DIM, activation = 'sigmoid', name = 'dec_output')(x)

        return Model(latent_inputs, decoder_outputs, name = 'decoder')

    def call(self, inputs, training = None):
        z_mean, z_log_var, z = self.encoder(inputs, training = training)
        reconstruction = self.decoder(z, training = training)

        # Input and output are now the same (no length field)
        target = inputs

        eps = 1e-7

        # BCE loss for 6-mers, 5-mers, 4-mers (scaled 100x)
        pred_clipped = ops.clip(reconstruction, eps, 1.0 - eps)
        target_6 = target[:, KMER_6_SLICE[0]:KMER_6_SLICE[1]]
        pred_6 = pred_clipped[:, KMER_6_SLICE[0]:KMER_6_SLICE[1]]
        bce_6 = -ops.mean(target_6 * ops.log(pred_6) + (1.0 - target_6) * ops.log(1.0 - pred_6))

        # MSE loss for 5-mers in log-space (add 0.01 offset)
        target_5 = target[:, KMER_5_SLICE[0]:KMER_5_SLICE[1]]
        pred_5 = reconstruction[:, KMER_5_SLICE[0]:KMER_5_SLICE[1]]
        mse_5 = ops.mean(ops.square(ops.log(target_5 + 0.01) - ops.log(pred_5 + 0.01)))

        # MSE loss for 4-mers in log-space (add 0.01 offset)
        target_4 = target[:, KMER_4_SLICE[0]:KMER_4_SLICE[1]]
        pred_4 = reconstruction[:, KMER_4_SLICE[0]:KMER_4_SLICE[1]]
        mse_4 = ops.mean(ops.square(ops.log(target_4 + 0.01) - ops.log(pred_4 + 0.01)))

        # MSE loss for 3-mers in log-space (add 0.01 offset)
        target_3 = target[:, KMER_3_SLICE[0]:KMER_3_SLICE[1]]
        pred_3 = reconstruction[:, KMER_3_SLICE[0]:KMER_3_SLICE[1]]
        mse_3 = ops.mean(ops.square(ops.log(target_3 + 0.01) - ops.log(pred_3 + 0.01)))

        # MSE loss for GC using logit transform: log(x / (1-x))
        eps = 1e-7
        target_gc = ops.clip(target[:, GC_SLICE[0]:GC_SLICE[1]], eps, 1.0 - eps)
        pred_gc = ops.clip(reconstruction[:, GC_SLICE[0]:GC_SLICE[1]], eps, 1.0 - eps)
        mse_gc = ops.mean(ops.square(ops.log(target_gc / (1.0 - target_gc)) - ops.log(pred_gc / (1.0 - pred_gc))))

        # Combined loss: BCE for 6-mers, MSE for 5/4/3-mers and GC
        recon_loss = bce_6 * OUTPUT_DIM * 100 + (mse_5 + mse_4 + mse_3 + mse_gc) * OUTPUT_DIM * 100 / 4

        # KL divergence
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


def load_data(file_path):
    """Load k-mer frequency data, skipping the length column.

    Args:
        file_path: Path to the .npy file containing frequency data.
            Expected format: length(1) + 6-mers(2080) + 5-mers(512) + 4-mers(136) + 3-mers(32) + GC(1) = 2,762 columns

    Returns:
        Data as float32 array with 2,761 features (length column removed).

    Raises:
        ValueError: If the data dimensions are unexpected.
    """
    logger.info(f'Loading data from {file_path}...')

    # Memory-map the file to avoid loading entire array into RAM at once
    data = np.load(file_path)

    # Validate input dimensions
    if data.ndim != 2:
        raise ValueError(f'Expected 2D array, got {data.ndim}D array with shape {data.shape}')
    if data.shape[1] != 2762:
        raise ValueError(f'Expected 2762 features (with length), got {data.shape[1]}. '
                         f'Data shape: {data.shape}')

    logger.info(f'Found {len(data)} samples, skipping length column...')

    # Skip column 0 (length) and convert to float32
    data_f32 = data[:, 1:].astype(np.float32)

    logger.info(f'Loaded {len(data_f32):,} samples with {data_f32.shape[1]} features')
    logger.info(f'Data stats: Min {data_f32.min():.6f}, Max {data_f32.max():.6f}, Mean {data_f32.mean():.6f}')
    return data_f32


def main():
    parser = argparse.ArgumentParser(description = 'Train a VAE on k-mer frequency data')
    parser.add_argument('-i', '--input', required = True, help = 'Path to input .npy file with k-mer frequencies')
    parser.add_argument('-e', '--epochs', type = int, default = 100, help = 'Number of training epochs (default: 100)')
    parser.add_argument('-l', '--learning-rate', type = float, default = 1e-4, help = 'Learning rate (default: 1e-4)')
    parser.add_argument('-b', '--batch-size', type = int, default = 4096, help = 'Batch size (default: 4096)')
    args = parser.parse_args()

    np.random.seed(SEED)
    keras.utils.set_random_seed(SEED)

    # Load data (no log transform for BCE)
    X = load_data(args.input)

    # Split
    X_train, X_val = train_test_split(X, test_size = 0.1, random_state = SEED)
    logger.info(f'Training on {X_train.shape[0]} samples, validating on {X_val.shape[0]} samples')

    # Load existing model if available, otherwise create new one
    resuming = os.path.exists('vae_best.keras')
    initial_best = None

    if resuming:
        logger.info('Loading existing model from vae_best.keras...')
        vae = keras.models.load_model('vae_best.keras', custom_objects = {
            'VAE': VAE, 'Sampling': Sampling, 'ClipLayer': ClipLayer,
            'SliceLayer': SliceLayer
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
    vae.compile(optimizer = keras.optimizers.Adam(learning_rate = args.learning_rate))

    # Setup callbacks
    vae_metrics = VAEMetricsCallback(validation_data = (X_val, X_val), sample_size = 5000)

    callbacks = [
        KLWarmupCallback(warmup_epochs = 5, max_weight = 0.1, skip_warmup = resuming),
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
        X_train,
        epochs = args.epochs,
        batch_size = args.batch_size,
        validation_data = (X_val, X_val),
        callbacks = callbacks,
        verbose = 0
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
