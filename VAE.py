#!/usr/bin/env python3
"""
Log-Space Variational Autoencoder for multi-scale k-mer frequency distributions.

Handles multi-dimensional inputs (INPUT_DIM features): sequence length, 7-mer, 4-mer,
3-mer frequencies, and GC content. Optimized for Keras 3 / JAX with 2M+ sequences.
"""

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
INPUT_DIM = 8362  # sequence length + 7-mer + 4-mer + 3-mer frequencies + GC content
LATENT_DIM = 256
SEED = 42


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

        z_mean, z_log_var, _ = self.model.encoder(sample_x, training = False)
        kl_loss = -0.5 * float(ops.mean(ops.sum(1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var), axis = 1)))  # type: ignore[arg-type]
        weighted_kl = float(self.model.kl_weight) * kl_loss

        predictions = self.model(sample_x, training = False)
        recon_loss = float(ops.mean(ops.sum(ops.square(sample_x - predictions), axis = 1)))  # type: ignore[arg-type]

        val_loss = logs.get('val_loss')
        val_loss_str = f'{val_loss:.2f}' if val_loss is not None else 'N/A'
        total = recon_loss + weighted_kl
        logger.info(f'Epoch {epoch + 1}/{self.params["epochs"]}: MSE Recon: {recon_loss:.2f}, KL: {kl_loss:.2f}, Weighted KL: {weighted_kl:.2f}, Val Loss: {val_loss_str}, Total: {total:.2f}')


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
        encoder_inputs = keras.Input(shape = (INPUT_DIM,))
        x = layers.Dense(4096)(encoder_inputs)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(negative_slope = 0.2)(x)
        x = layers.Dense(2048)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(negative_slope = 0.2)(x)
        x = layers.Dense(1024)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(negative_slope = 0.2)(x)

        z_mean = layers.Dense(self.latent_dim, name = 'z_mean')(x)
        z_log_var = layers.Dense(self.latent_dim, name = 'z_log_var')(x)
        z_log_var = ClipLayer(name = 'z_log_var_clip')(z_log_var)
        z = Sampling()([z_mean, z_log_var])

        return Model(encoder_inputs, [z_mean, z_log_var, z], name = 'encoder')

    def _build_decoder(self):
        latent_inputs = keras.Input(shape = (self.latent_dim,))

        x = layers.Dense(1024)(latent_inputs)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(negative_slope = 0.2)(x)
        x = layers.Dense(2048)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(negative_slope = 0.2)(x)
        x = layers.Dense(4096)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(negative_slope = 0.2)(x)
        decoder_outputs = layers.Dense(INPUT_DIM, activation = 'linear')(x)  # Linear activation for log-space prediction

        return Model(latent_inputs, decoder_outputs, name = 'decoder')

    def call(self, inputs, training = None):
        z_mean, z_log_var, z = self.encoder(inputs, training = training)
        reconstruction = self.decoder(z, training = training)

        recon_loss = ops.mean(ops.sum(ops.square(inputs - reconstruction), axis = 1))  # Sum over features

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

    data = np.load(file_path)

    # Validate input dimensions
    if data.ndim != 2:
        raise ValueError(f'Expected 2D array, got {data.ndim}D array with shape {data.shape}')
    if data.shape[1] != expected_dim:
        raise ValueError(f'Expected {expected_dim} features, got {data.shape[1]}. '
                         f'Data shape: {data.shape}')

    logger.info('Transforming data to Log-Space (Log(x + 1e-6))...')
    data_log = np.log(data.astype(np.float32) + 1e-6)

    logger.info(f'Loaded {len(data_log)} samples')
    logger.info(f'Data stats: Min {data_log.min():.2f}, Max {data_log.max():.2f}, Mean {data_log.mean():.2f}')
    return data_log


def main():
    np.random.seed(SEED)
    keras.utils.set_random_seed(SEED)

    # Load Log-Transformed Data
    data_path = './Data/all_multimer_frequencies_l5000_shuffled.npy'
    X = load_data_log_space(data_path)

    # Split
    X_train, X_val = train_test_split(X, test_size = 0.2, random_state = SEED)
    logger.info(f'Training on {X_train.shape[0]} samples, validating on {X_val.shape[0]} samples')

    # Load existing model if available, otherwise create new one
    resuming = os.path.exists('vae_best.keras')
    initial_best = None

    if resuming:
        logger.info('Loading existing model from vae_best.keras...')
        vae = keras.models.load_model('vae_best.keras', custom_objects = {'VAE': VAE, 'Sampling': Sampling, 'ClipLayer': ClipLayer})
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
    vae.compile(optimizer = keras.optimizers.Adam(learning_rate = 1e-4))  # type: ignore[arg-type]

    # Setup callbacks
    vae_metrics = VAEMetricsCallback(validation_data = (X_val, X_val), sample_size = 5000)

    callbacks = [
        KLWarmupCallback(warmup_epochs = 5, max_weight = 1.0, skip_warmup = resuming),
        vae_metrics,
        VAECheckpoint(filepath_prefix = 'vae', monitor = 'val_loss', verbose = 0, initial_best = initial_best)
    ]

    logger.info('Starting Training...')
    history = vae.fit(
        X_train,
        X_train,  # Input = Output (Log Space)
        epochs = 500,
        batch_size = 4096,  # Large batch size for stability
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
