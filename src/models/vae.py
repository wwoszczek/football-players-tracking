from __future__ import annotations

from typing import Union

import tensorflow as tf


class VAE(tf.keras.Model):
    def __init__(
        self,
        encoder: tf.keras.Model,
        decoder: tf.keras.Model,
        reconstruction_loss_weight: Union[int, float] = 100,
        kl_loss_weight: Union[int, float] = 1,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.decoder = decoder

        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.kl_loss_weight = kl_loss_weight

    def train_step(self, data: tf.Tensor) -> dict[str, tf.Tensor]:
        with tf.GradientTape() as tape:
            losses = self._calculate_losses(data)

        grads = tape.gradient(losses["loss"], self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return losses

    def test_step(self, data: tf.Tensor) -> dict[str, tf.Tensor]:
        return self._calculate_losses(data)

    def _calculate_losses(self, data: tf.Tensor) -> dict[str, tf.Tensor]:
        z_mean, z_log_var, z = self.encoder(data)

        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(tf.keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2))
        )

        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

        total_loss = self.reconstruction_loss_weight * reconstruction_loss + self.kl_loss_weight * kl_loss

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def call(
        self, inputs: tf.Tensor, training: bool = None, mask: None | tf.Tensor | list[None | tf.Tensor] = None
    ) -> tf.Tensor:
        _, _, z = self.encoder(inputs)
        return self.decoder(z)
