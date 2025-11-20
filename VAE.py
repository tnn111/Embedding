#!/usr/bin/env python3
"""
Variational Autoencoder for 7-mer frequency distributions.
Uses Keras 3 with JAX backend.
"""

import os
os.environ['KERAS_BACKEND'] = 'jax'

# Configure JAX for optimal GPU usage
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # Disable memory preallocation for better memory management
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'  # Use platform allocator

import numpy as np
import jax
import keras
from keras import layers, Model, ops
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import pickle


class KLWarmupCallback(keras.callbacks.Callback):
    """Gradually increase KL loss weight during training to prevent posterior collapse"""

    def __init__(self, warmup_epochs = 10, max_weight = 1.0):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.max_weight = max_weight

    def on_epoch_begin(self, epoch, logs = None):
        if epoch < self.warmup_epochs:
            new_weight = (epoch / self.warmup_epochs) * self.max_weight
        else:
            new_weight = self.max_weight

        self.model.kl_weight = new_weight
        print(f'KL weight: {new_weight:.4f}')


class Sampling(layers.Layer):
    """Reparameterization trick: sample z = mean + exp(0.5 * log_var) * epsilon"""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape = (batch, dim))
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon


class VAE(Model):
    """Variational Autoencoder for k-mer frequency distributions"""

    def __init__(self, latent_dim = 128, kl_weight = 1.0, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight

        # Encoder
        self.encoder = self._build_encoder()

        # Decoder
        self.decoder = self._build_decoder()

        # Metrics
        self.total_loss_tracker = keras.metrics.Mean(name = 'total_loss')
        self.reconstruction_loss_tracker = keras.metrics.Mean(name = 'reconstruction_loss')
        self.kl_loss_tracker = keras.metrics.Mean(name = 'kl_loss')
        self.kl_weight_tracker = keras.metrics.Mean(name = 'kl_weight')

    def _build_encoder(self):
        """Build encoder: 8192 -> 2048 -> 512 -> 128 (latent) with batch normalization"""
        encoder_inputs = keras.Input(shape = (8192,))

        # First layer
        x = layers.Dense(2048)(encoder_inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        # Second layer
        x = layers.Dense(512)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        # Latent space parameters
        z_mean = layers.Dense(self.latent_dim, name = 'z_mean')(x)
        z_log_var = layers.Dense(self.latent_dim, name = 'z_log_var')(x)

        # Clip log variance to prevent numerical instability
        # Range [-20, 2] corresponds to variance in [e^-20, e^2] ≈ [2e-9, 7.4]
        # This prevents both explosion (too large variance) and collapse (too small variance)
        z_log_var = ops.clip(z_log_var, -20, 2)

        z = Sampling()([z_mean, z_log_var])

        return Model(encoder_inputs, [z_mean, z_log_var, z], name = 'encoder')

    def _build_decoder(self):
        """Build decoder: 128 (latent) -> 512 -> 2048 -> 8192 with batch normalization"""
        latent_inputs = keras.Input(shape = (self.latent_dim,))

        # First layer
        x = layers.Dense(512)(latent_inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        # Second layer
        x = layers.Dense(2048)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        # Output layer with softmax for probability distribution
        decoder_outputs = layers.Dense(8192, activation = 'softmax')(x)

        return Model(latent_inputs, decoder_outputs, name = 'decoder')

    def call(self, inputs, training = None):
        """Forward pass through the VAE"""
        _, _, z = self.encoder(inputs, training = training)
        reconstructed = self.decoder(z, training = training)
        return reconstructed

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.kl_weight_tracker,
        ]

    def train_step(self, data):
        """Custom training step with VAE loss"""

        # Define loss computation inside a function for gradient computation
        def compute_loss_and_forward():
            z_mean, z_log_var, z = self.encoder(data, training = True)
            reconstruction = self.decoder(z, training = True)

            # Reconstruction loss using categorical cross-entropy
            reconstruction_loss = ops.mean(
                keras.losses.categorical_crossentropy(data, reconstruction)
            )

            # KL divergence loss
            kl_loss = -0.5 * ops.mean(
                ops.sum(
                    1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var),
                    axis = 1
                )
            )

            # Total loss with KL weight
            total_loss = reconstruction_loss + self.kl_weight * kl_loss

            return total_loss, reconstruction_loss, kl_loss

        # Compute gradients
        grads = self.optimizer.compute_gradients(
            lambda: compute_loss_and_forward()[0],
            self.trainable_weights
        )
        self.optimizer.apply_gradients(grads)

        # Recompute losses for metrics (necessary with current Keras API)
        total_loss, reconstruction_loss, kl_loss = compute_loss_and_forward()

        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.kl_weight_tracker.update_state(self.kl_weight)

        return {
            'total_loss': self.total_loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result(),
            'kl_weight': self.kl_weight_tracker.result(),
        }

    def test_step(self, data):
        """Custom test step"""
        # Forward pass (use inference mode for batch normalization)
        z_mean, z_log_var, z = self.encoder(data, training = False)
        reconstruction = self.decoder(z, training = False)

        # Reconstruction loss using categorical cross-entropy
        reconstruction_loss = ops.mean(
            keras.losses.categorical_crossentropy(data, reconstruction)
        )

        # KL divergence loss
        kl_loss = -0.5 * ops.mean(
            ops.sum(
                1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var),
                axis = 1
            )
        )

        # Total loss with KL weight
        total_loss = reconstruction_loss + self.kl_weight * kl_loss

        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.kl_weight_tracker.update_state(self.kl_weight)

        return {
            'total_loss': self.total_loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result(),
            'kl_weight': self.kl_weight_tracker.result(),
        }


def load_data(file_path):
    """Load k-mer frequency data from file with validation"""
    print(f'Loading data from {file_path}...')
    data = np.loadtxt(file_path, dtype = np.float32)

    # Validate data shape
    assert data.ndim == 2, f'Expected 2D array, got {data.ndim}D'
    assert data.shape[1] == 8192, f'Expected 8192 features (4^7 k-mers), got {data.shape[1]}'

    # Check if data represents probability distributions
    row_sums = data.sum(axis = 1)
    mean_sum = row_sums.mean()
    print(f'Loaded {data.shape[0]} sequences with {data.shape[1]} features')
    print(f'Mean row sum: {mean_sum:.6f} (should be ~1.0 for probability distributions)')
    print(f'Row sum range: [{row_sums.min():.6f}, {row_sums.max():.6f}]')

    # Warn if data doesn't look like probabilities
    if not (0.99 < mean_sum < 1.01):
        print('WARNING: Data does not appear to be normalized (rows should sum to 1)')

    return data


def main():
    # Print JAX device information
    print('=' * 60)
    print('JAX Configuration:')
    print(f'JAX version: {jax.__version__}')
    print(f'Available devices: {jax.devices()}')
    print(f'Default backend: {jax.default_backend()}')
    print(f'Keras backend: {keras.backend.backend()}')
    print('=' * 60)
    print()

    # Set random seeds for reproducibility
    np.random.seed(42)
    keras.utils.set_random_seed(42)  # This also sets JAX seed when using JAX backend

    # Load data
    data_path = './Data/kmer_frequencies_l5000_shuffled.txt'
    X = load_data(data_path)

    # Split into train and validation sets (80/20)
    X_train, X_val = train_test_split(X, test_size = 0.2, random_state = 42)
    print(f'Training set: {X_train.shape[0]} sequences')
    print(f'Validation set: {X_val.shape[0]} sequences')

    # Create VAE model with initial KL weight of 0 (will be increased by warmup callback)
    latent_dim = 128
    vae = VAE(latent_dim = latent_dim, kl_weight = 0.0)

    # Compile model
    vae.compile(optimizer = keras.optimizers.Adam(learning_rate = 1e-3))  # type: ignore

    # Callbacks
    callbacks = [
        KLWarmupCallback(warmup_epochs = 10, max_weight = 1.0),
        EarlyStopping(
            monitor = 'val_total_loss',
            patience = 15,
            restore_best_weights = True,
            verbose = 1
        ),
        ReduceLROnPlateau(
            monitor = 'val_total_loss',
            factor = 0.5,
            patience = 5,
            min_lr = 1e-6,
            verbose = 1
        ),
        ModelCheckpoint(
            'vae_best_model.keras',
            monitor = 'val_total_loss',
            save_best_only = True,
            verbose = 1
        )
    ]

    # Train the model
    print('\nTraining VAE...')
    history = vae.fit(
        X_train,
        epochs = 100,
        batch_size = 256,
        validation_data = X_val,
        callbacks = callbacks,
        verbose = 1  # type: ignore
    )

    # Save final model
    vae.save('vae_final_model.keras')
    print('\nModel saved to vae_final_model.keras')

    # Save encoder and decoder separately for easier use
    vae.encoder.save('vae_encoder.keras')
    vae.decoder.save('vae_decoder.keras')
    print('Encoder saved to vae_encoder.keras')
    print('Decoder saved to vae_decoder.keras')

    # Save training history
    with open('vae_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    print('Training history saved to vae_history.pkl')

    # Print final metrics
    print(f'\nFinal training loss: {history.history["total_loss"][-1]:.4f}')
    print(f'Final validation loss: {history.history["val_total_loss"][-1]:.4f}')
    print(f'Final reconstruction loss: {history.history["reconstruction_loss"][-1]:.4f}')
    print(f'Final KL loss: {history.history["kl_loss"][-1]:.4f}')


if __name__ == '__main__':
    main()
