#!/usr/bin/env python3
"""
Variational Autoencoder for 7-mer frequency distributions using 1D convolutions.

Input: 7-mer frequencies (8,192 canonical k-mers) as a 1D signal.
Output: Reconstructed 7-mer frequencies with sigmoid activation.
Loss: BCE (binary cross-entropy) for reconstruction.

Architecture uses two Conv1D layers with three pooling stages to compress
the signal before a dense layer leading to the 256-dimensional latent space.
"""

import argparse
import os
os.environ['KERAS_BACKEND'] = 'jax'

import logging
import numpy as np
import keras
from keras import layers
from keras import Model
from keras import ops
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
INPUT_DIM = 8192  # 7-mer canonical frequencies
LATENT_DIM = 256
SEED = 42


class VAEMetricsCallback(keras.callbacks.Callback):
    """Track VAE metrics outside of JIT-compiled code."""

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
        target = sample_x

        # BCE loss
        eps = 1e-7
        target_clipped = ops.clip(target, eps, 1.0 - eps)
        pred_clipped = ops.clip(reconstruction, eps, 1.0 - eps)
        bce = float(-ops.mean(target_clipped * ops.log(pred_clipped) + (1.0 - target_clipped) * ops.log(1.0 - pred_clipped)))

        recon_loss = bce * INPUT_DIM

        val_loss = logs.get('val_loss')
        val_loss_str = f'{val_loss:.2f}' if val_loss is not None else 'N/A'
        logger.info(
            f'Epoch {epoch + 1}/{self.params["epochs"]}: Recon: {recon_loss:.2f}, KL: {kl_loss:.2f} (w={weighted_kl:.2f}), Val: {val_loss_str}, BCE: {bce:.6f}'
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


class VAE(Model):
    def __init__(self, latent_dim = LATENT_DIM, kl_weight = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.kl_weight = keras.Variable(kl_weight, trainable = False, dtype = 'float32', name = 'kl_weight')
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

    def _build_encoder(self):
        """Build fully-connected encoder (no convolutions to avoid CUDA issues).

        Architecture:
            Input (8192,) flattened
            → Dense(1024) → Dense(512) → Dense(256) z_mean, z_log_var
        """
        encoder_inputs = keras.Input(shape = (INPUT_DIM, 1), name = 'encoder_input')

        # Flatten input
        x = layers.Flatten(name = 'enc_flatten')(encoder_inputs)

        # Dense layers
        x = layers.Dense(1024, name = 'enc_dense1')(x)
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
        """Build fully-connected decoder (mirror of encoder).

        Architecture:
            Input (256)
            → Dense(512) → Dense(1024) → Dense(8192) sigmoid
            → Output (8192, 1)
        """
        latent_inputs = keras.Input(shape = (self.latent_dim,), name = 'decoder_input')

        # Dense layers
        x = layers.Dense(512, name = 'dec_dense1')(latent_inputs)
        x = layers.BatchNormalization(name = 'dec_bn1')(x)
        x = layers.LeakyReLU(negative_slope = 0.2, name = 'dec_relu1')(x)

        x = layers.Dense(1024, name = 'dec_dense2')(x)
        x = layers.BatchNormalization(name = 'dec_bn2')(x)
        x = layers.LeakyReLU(negative_slope = 0.2, name = 'dec_relu2')(x)

        # Output layer
        x = layers.Dense(INPUT_DIM, activation = 'sigmoid', name = 'dec_output')(x)

        # Reshape to match input shape
        decoder_outputs = layers.Reshape((INPUT_DIM, 1), name = 'dec_reshape')(x)

        return Model(latent_inputs, decoder_outputs, name = 'decoder')

    def call(self, inputs, training = None):
        z_mean, z_log_var, z = self.encoder(inputs, training = training)
        reconstruction = self.decoder(z, training = training)

        # BCE loss
        eps = 1e-7
        target = ops.clip(inputs, eps, 1.0 - eps)
        pred = ops.clip(reconstruction, eps, 1.0 - eps)
        bce = -ops.mean(target * ops.log(pred) + (1.0 - target) * ops.log(1.0 - pred))
        recon_loss = bce * INPUT_DIM

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


def load_data_to_memory(file_path: str, start_idx: int, end_idx: int) -> np.ndarray:
    """Load a subset of the data into memory as float32.

    This avoids memory leaks caused by repeated numpy->TensorFlow conversions
    in the training loop.

    Args:
        file_path: Path to the .npy file
        start_idx: First row index to load
        end_idx: Last row index (exclusive)

    Returns:
        numpy array of shape (n_samples, INPUT_DIM, 1) as float32
    """
    logger.info(f'Loading data[{start_idx}:{end_idx}] into memory...')
    data_mmap = np.load(file_path, mmap_mode = 'r')
    data = data_mmap[start_idx:end_idx, 1:8193].astype(np.float32).reshape(-1, INPUT_DIM, 1)
    logger.info(f'Loaded {data.shape[0]:,} samples ({data.nbytes / 1024**3:.1f} GB)')
    return data


class NumpyBatchDataset(keras.utils.PyDataset):
    """PyDataset that yields batches from pre-loaded numpy data.

    Works with JAX backend - no TensorFlow dependency.
    Uses indexed access (__getitem__) instead of iteration for Keras compatibility.
    """

    def __init__(self, data: np.ndarray, batch_size: int, shuffle: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(data)
        self.n_batches = self.n_samples // batch_size
        self.indices = np.arange(self.n_samples)
        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return self.n_batches

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = start + self.batch_size
        batch_indices = self.indices[start:end]
        batch = self.data[batch_indices]
        return batch, batch  # Autoencoder: input = target

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def get_data_info(file_path, max_samples = None):
    """Get data info and split ranges without loading data.

    Uses contiguous ranges for train/val to enable efficient sequential reads.
    The file should already be shuffled.

    Args:
        file_path: Path to the .npy file.
        max_samples: If set, only use this many samples.

    Returns:
        Tuple of (n_samples, train_end, val_start, val_end).
    """
    logger.info(f'Reading data info from {file_path}...')

    # Memory-map to get shape without loading
    data = np.load(file_path, mmap_mode = 'r')

    if data.ndim != 2:
        raise ValueError(f'Expected 2D array, got {data.ndim}D array with shape {data.shape}')

    n_total = len(data)
    logger.info(f'Found {n_total:,} samples with {data.shape[1]} features')

    # Limit samples if requested
    if max_samples is not None and max_samples < n_total:
        logger.info(f'Limiting to {max_samples:,} samples')
        n_total = max_samples

    # Use first 90% for train, last 10% for val (contiguous ranges)
    train_end = int(n_total * 0.9)
    val_start = train_end
    val_end = n_total

    logger.info(f'Train: 0-{train_end:,} ({train_end:,}), Val: {val_start:,}-{val_end:,} ({val_end - val_start:,})')
    return n_total, train_end, val_start, val_end


def main():
    parser = argparse.ArgumentParser(description = 'Train a convolutional VAE on 7-mer frequency data')
    parser.add_argument('-i', '--input', required = True, help = 'Path to input .npy file with k-mer frequencies')
    parser.add_argument('-e', '--epochs', type = int, default = 100, help = 'Number of training epochs (default: 100)')
    parser.add_argument('-l', '--learning-rate', type = float, default = 1e-4, help = 'Learning rate (default: 1e-4)')
    parser.add_argument('-b', '--batch-size', type = int, default = 1024, help = 'Batch size (default: 1024)')
    parser.add_argument('-n', '--max-samples', type = int, default = None, help = 'Max samples to load (for testing)')
    args = parser.parse_args()

    np.random.seed(SEED)
    keras.utils.set_random_seed(SEED)

    # Get data info and split ranges (no loading)
    n_samples, train_end, val_start, val_end = get_data_info(args.input, max_samples = args.max_samples)

    # Load existing model if available, otherwise create new one
    resuming = os.path.exists('vae_best.keras')
    initial_best = None

    if resuming:
        logger.info('Loading existing model from vae_best.keras...')
        vae = keras.models.load_model('vae_best.keras', custom_objects = {
            'VAE': VAE, 'Sampling': Sampling, 'ClipLayer': ClipLayer
        })
        logger.info('Model loaded successfully. Recompiling and continuing training...')
        if os.path.exists('vae_history.pkl'):
            with open('vae_history.pkl', 'rb') as f:
                prev_history = pickle.load(f)
                if 'val_loss' in prev_history and prev_history['val_loss']:
                    initial_best = min(prev_history['val_loss'])
                    logger.info(f'Previous best val_loss: {initial_best:.4f}')
    else:
        logger.info('No existing model found. Creating new VAE...')
        vae = VAE(latent_dim = LATENT_DIM)

    # Print model summary
    vae.encoder.summary(print_fn = logger.info)
    vae.decoder.summary(print_fn = logger.info)

    logger.info('Compiling model...')

    # Always (re)compile to ensure consistent optimizer settings
    vae.compile(optimizer = keras.optimizers.Adam(learning_rate = args.learning_rate))

    logger.info('Model compiled successfully.')

    # Load all data into memory
    # For 4.8M samples: ~157GB RAM needed
    logger.info('Loading training data into memory...')
    train_data = load_data_to_memory(args.input, 0, train_end)

    logger.info('Loading validation data into memory...')
    val_data = load_data_to_memory(args.input, val_start, val_end)

    # Create batch datasets (works with JAX backend)
    logger.info('Creating batch datasets...')
    train_dataset = NumpyBatchDataset(train_data, args.batch_size, shuffle = True)
    val_dataset = NumpyBatchDataset(val_data, args.batch_size, shuffle = False)

    # Use a subset of validation data for the metrics callback
    val_sample = val_data[:min(5000, len(val_data))]
    logger.info(f'Validation sample: {val_sample.shape}')

    # Setup callbacks
    vae_metrics = VAEMetricsCallback(validation_data = (val_sample, val_sample), sample_size = 5000)

    callbacks = [
        KLWarmupCallback(warmup_epochs = 5, max_weight = 0.1, skip_warmup = resuming),
        vae_metrics,
        VAECheckpoint(filepath_prefix = 'vae', monitor = 'val_loss', verbose = 1, initial_best = initial_best),
        keras.callbacks.ReduceLROnPlateau(
            monitor = 'val_loss',
            factor = 0.5,
            patience = 20,
            min_lr = 1e-6,
            verbose = 1
        )
    ]

    n_train_batches = len(train_data) // args.batch_size
    n_val_batches = len(val_data) // args.batch_size
    logger.info(f'Train batches: {n_train_batches}, Val batches: {n_val_batches}')
    logger.info('Starting Training...')
    history = vae.fit(
        train_dataset,
        epochs = args.epochs,
        validation_data = val_dataset,
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
