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
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split


class Sampling(layers.Layer):
    """Reparameterization trick: sample z = mean + exp(0.5 * log_var) * epsilon"""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim))
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon


class VAE(Model):
    """Variational Autoencoder for k-mer frequency distributions"""

    def __init__(self, latent_dim=128, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = self._build_encoder()

        # Decoder
        self.decoder = self._build_decoder()

        # Metrics
        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
        self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')
        self.kl_loss_tracker = keras.metrics.Mean(name='kl_loss')

    def _build_encoder(self):
        """Build encoder: 8192 -> 2048 -> 512 -> 128 (latent)"""
        encoder_inputs = keras.Input(shape=(8192,))
        x = layers.Dense(2048, activation='relu')(encoder_inputs)
        x = layers.Dense(512, activation='relu')(x)
        z_mean = layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(x)
        z = Sampling()([z_mean, z_log_var])

        return Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')

    def _build_decoder(self):
        """Build decoder: 128 (latent) -> 512 -> 2048 -> 8192"""
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(512, activation='relu')(latent_inputs)
        x = layers.Dense(2048, activation='relu')(x)
        decoder_outputs = layers.Dense(8192, activation='softmax')(x)

        return Model(latent_inputs, decoder_outputs, name='decoder')

    def call(self, inputs):
        """Forward pass through the VAE"""
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        return reconstructed

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        """Custom training step with VAE loss"""
        # Forward pass
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)

        # Compute losses
        def compute_loss():
            # Reconstruction loss (cross-entropy)
            # Since we're dealing with probability distributions, use categorical cross-entropy
            reconstruction_loss = ops.mean(
                ops.sum(
                    -data * ops.log(reconstruction + 1e-10),
                    axis=1
                )
            )

            # KL divergence loss
            kl_loss = -0.5 * ops.mean(
                ops.sum(
                    1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var),
                    axis=1
                )
            )

            # Total loss
            return reconstruction_loss + kl_loss, reconstruction_loss, kl_loss

        # Compute loss and gradients
        loss_fn = lambda: compute_loss()[0]
        grads = self.optimizer.compute_gradients(loss_fn, self.trainable_weights)
        self.optimizer.apply_gradients(grads)

        # Compute losses for metrics (without gradients)
        total_loss, reconstruction_loss, kl_loss = compute_loss()

        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            'total_loss': self.total_loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        """Custom test step"""
        # Forward pass
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)

        # Reconstruction loss
        reconstruction_loss = ops.mean(
            ops.sum(
                -data * ops.log(reconstruction + 1e-10),
                axis=1
            )
        )

        # KL divergence loss
        kl_loss = -0.5 * ops.mean(
            ops.sum(
                1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var),
                axis=1
            )
        )

        # Total loss
        total_loss = reconstruction_loss + kl_loss

        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            'total_loss': self.total_loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result(),
        }


def load_data(file_path):
    """Load k-mer frequency data from file"""
    print(f'Loading data from {file_path}...')
    data = np.loadtxt(file_path, dtype=np.float32)
    print(f'Loaded {data.shape[0]} sequences with {data.shape[1]} features')
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
    keras.utils.set_random_seed(42)

    # Load data
    data_path = './Data/kmer_frequencies_l5000_shuffled.txt'
    X = load_data(data_path)

    # Split into train and validation sets (80/20)
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
    print(f'Training set: {X_train.shape[0]} sequences')
    print(f'Validation set: {X_val.shape[0]} sequences')

    # Create VAE model
    latent_dim = 128
    vae = VAE(latent_dim=latent_dim)

    # Compile model
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_total_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            'vae_best_model.keras',
            monitor='val_total_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    # Train the model
    print('\nTraining VAE...')
    history = vae.fit(
        X_train,
        epochs=50,
        batch_size=256,
        validation_data=(X_val, X_val),
        callbacks=callbacks,
        verbose=1
    )

    # Save final model
    vae.save('vae_final_model.keras')
    print('\nModel saved to vae_final_model.keras')

    # Save encoder and decoder separately for easier use
    vae.encoder.save('vae_encoder.keras')
    vae.decoder.save('vae_decoder.keras')
    print('Encoder saved to vae_encoder.keras')
    print('Decoder saved to vae_decoder.keras')

    # Print final metrics
    print(f'\nFinal training loss: {history.history["total_loss"][-1]:.4f}')
    print(f'Final validation loss: {history.history["val_total_loss"][-1]:.4f}')


if __name__ == '__main__':
    main()
