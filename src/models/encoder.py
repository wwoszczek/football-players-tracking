import tensorflow as tf
from .blocks import ConvBlock


class Sampling(tf.keras.layers.Layer):
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_encoder(
    image_shape: tuple[int],
    latent_dim: int,
    conv_block_filters: list[int],
    conv_block_kernel_sizes: list[int],
    conv_block_strides: list[int],
    conv_block_dropout_rates: list[float],
) -> tf.keras.Model:
    encoder_inputs = tf.keras.Input(shape=image_shape)

    x = ConvBlock(
        filters=conv_block_filters[0],
        kernel_size=conv_block_kernel_sizes[0],
        strides=conv_block_strides[0],
        dropout_rate=conv_block_dropout_rates[0],
    )(encoder_inputs)

    for conv_block_filter, conv_block_kernel_size, conv_block_stride, conv_block_dropout_rate in zip(
        conv_block_filters[1:], conv_block_kernel_sizes[1:], conv_block_strides[1:], conv_block_dropout_rates[1:]
    ):
        x = ConvBlock(
            filters=conv_block_filter,
            kernel_size=conv_block_kernel_size,
            strides=conv_block_stride,
            dropout_rate=conv_block_dropout_rate,
        )(x)

    x = tf.keras.layers.Flatten()(x)

    z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])

    return tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
