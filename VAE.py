#!/usr/bin/env python3
"""
Log-Space Variational Autoencoder for 7-mer frequency distributions.
Optimized for Keras 3 / JAX with 2M+ sequences.
"""

import os
os.environ['KERAS_BACKEND'] = 'jax'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import numpy as np
import keras
from keras import layers, Model, ops
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import pickle

# --- 1. CUSTOM CALLBACKS ---
class VAEMetricsCallback(keras.callbacks.Callback):
    """Track VAE metrics outside of JIT-compiled code (Keras 3 compatible)"""

    def __init__(self):
        super().__init__()
        self.validation_data: tuple | None = None

    def on_epoch_end(self, epoch, logs=None):
        if logs is None or self.validation_data is None:
            return

        sample_idx = np.random.choice(len(self.validation_data[0]), min(1000, len(self.validation_data[0])), replace=False)
        sample_x = self.validation_data[0][sample_idx]

        # Get encoder outputs
        z_mean, z_log_var, _ = self.model.encoder(sample_x, training=False)

        # KL loss
        kl_loss = -0.5 * float(ops.mean(  # type: ignore[arg-type]
            ops.sum(1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var), axis=1)
        ))
        weighted_kl = float(self.model.kl_weight) * kl_loss

        # Reconstruction loss (MSE)
        predictions = self.model(sample_x, training=False)
        recon_loss = float(ops.mean(ops.sum(ops.square(sample_x - predictions), axis=1)))  # type: ignore[arg-type]

        val_loss = logs.get('val_loss', 0)
        total = recon_loss + weighted_kl
        print(f'Epoch {epoch + 1}/{self.params["epochs"]}: MSE Recon: {recon_loss:.2f}, KL: {kl_loss:.2f}, Weighted KL: {weighted_kl:.2f}, Val Loss: {val_loss:.2f}, Total: {total:.2f}')


class KLWarmupCallback(keras.callbacks.Callback):
    def __init__(self, warmup_epochs=10, max_weight=1.0):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.max_weight = max_weight

    def on_epoch_begin(self, epoch, _logs=None):
        if epoch < self.warmup_epochs:
            new_weight = (epoch / self.warmup_epochs) * self.max_weight
        else:
            new_weight = self.max_weight
        self.model.kl_weight.assign(new_weight)

# --- 2. SAMPLING LAYER ---
class Sampling(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon

    def get_config(self):
        return super().get_config()

# --- 3. VAE MODEL (LOG SPACE) ---
class VAE(Model):
    def __init__(self, latent_dim=128, kl_weight=1.0, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.kl_weight = keras.Variable(kl_weight, trainable=False, dtype='float32', name='kl_weight')
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

    def _build_encoder(self):
        encoder_inputs = keras.Input(shape=(8192,))
        
        # Use LeakyReLU for better gradient flow with Log data
        x = layers.Dense(2048)(encoder_inputs)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(negative_slope=0.2)(x) 
        
        x = layers.Dense(512)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(negative_slope=0.2)(x)
        
        z_mean = layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(x)
        z_log_var = ops.clip(z_log_var, -20, 2)
        
        z = Sampling()([z_mean, z_log_var])
        return Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')

    def _build_decoder(self):
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        
        x = layers.Dense(512)(latent_inputs)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(negative_slope=0.2)(x)
        
        x = layers.Dense(2048)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(negative_slope=0.2)(x)
        
        # IMPORTANT: Linear activation because we are predicting Log values (range -inf to +inf)
        # Do NOT use Softmax here.
        decoder_outputs = layers.Dense(8192, activation='linear')(x)
        
        return Model(latent_inputs, decoder_outputs, name='decoder')

    def call(self, inputs, training=None):
        z_mean, z_log_var, z = self.encoder(inputs, training=training)
        reconstruction = self.decoder(z, training=training)

        # --- LOSS CALCULATION ---
        
        # 1. Reconstruction Loss: MSE (Standard for Log-Space data)
        # We sum over the 8192 dimensions to keep the magnitude relevant compared to KL
        recon_loss = ops.mean(ops.sum(ops.square(inputs - reconstruction), axis=1))
        
        # 2. KL Divergence
        kl_loss = -0.5 * ops.mean(
            ops.sum(1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var), axis=1)
        )

        # Total Loss (MSE naturally balances better with KL than Crossentropy does)
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

# --- 4. DATA LOADING (LOG TRANSFORM) ---
def load_data_log_space(file_path):
    print(f'Loading data from {file_path}...')
    
    # [MOCK DATA FOR DEMO - REPLACE WITH YOUR np.load]
    if not os.path.exists(file_path):
        print("Generating mock data...")
        data = np.random.dirichlet(alpha=np.ones(8192)*0.1, size=5000).astype('float32')
    else:
        data = np.load(file_path)

    # 1. Normalize to Probability Sum=1 (Safety check)
    row_sums = data.sum(axis=1, keepdims=True) + 1e-9
    data = data / row_sums
    
    # 2. LOG TRANSFORM
    # Add small epsilon to avoid log(0). 1e-6 is standard for k-mer freqs.
    print("Transforming data to Log-Space (Log(x + 1e-6))...")
    data_log = np.log(data + 1e-6)
    
    print(f"Data stats: Min {data_log.min():.2f}, Max {data_log.max():.2f}, Mean {data_log.mean():.2f}")
    return data_log

def main():
    np.random.seed(42)
    keras.utils.set_random_seed(42)

    # Load Log-Transformed Data
    data_path = './Data/kmer_frequencies_l5000_shuffled.npy'
    X = load_data_log_space(data_path)

    # Split
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
    print(f'Training on {X_train.shape[0]} sequences')

    # Model Setup
    vae = VAE(latent_dim=128, kl_weight=0.0)

    # NOTE: learning_rate=1e-3 is safer for MSE. 
    # If loss is too high/NaN, lower to 1e-4.
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))  # type: ignore[arg-type] 

    # Setup metrics callback
    vae_metrics = VAEMetricsCallback()
    vae_metrics.validation_data = (X_val, X_val)

    # Callbacks
    callbacks = [
        KLWarmupCallback(warmup_epochs=5, max_weight=1.0),
        vae_metrics,
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=0),
        ModelCheckpoint('vae_log_best.keras', monitor='val_loss', save_best_only=True, verbose=0)
    ]

    # Train - Larger Batch Size for 2M samples
    print('\nStarting Training...')
    history = vae.fit(
        X_train, 
        X_train, # Input = Output (Log Space)
        epochs=500,
        batch_size=1024, # Large batch size for stability
        validation_data=(X_val, X_val),
        callbacks=callbacks,
        verbose=0  # type: ignore[arg-type]
    )
    
    # Save
    vae.save('vae_log_final.keras')
    with open('vae_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)

if __name__ == '__main__':
    main()
