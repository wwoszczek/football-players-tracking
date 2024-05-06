import tensorflow as tf
from .blocks import ConvTBlock


def build_decoder(
    latent_dim: int,
    dense_layer_units: int,
    reshape_layer_target_shape: tuple[int],
    convt_block_filters: list[int],
    convt_block_kernel_sizes: list[int],
    convt_block_strides: list[int],
    convt_block_dropout_rates: list[float],
) -> tf.keras.Model:
    latent_inputs = tf.keras.layers.Input(shape=(latent_dim,))

    x = tf.keras.layers.Dense(dense_layer_units)(latent_inputs)
    x = tf.keras.layers.Reshape(reshape_layer_target_shape)(x)

    for convt_block_filter, convt_block_kernel_size, convt_block_stride, convt_block_dropout_rate in zip(
        convt_block_filters, convt_block_kernel_sizes, convt_block_strides, convt_block_dropout_rates
    ):
        x = ConvTBlock(
            filters=convt_block_filter,
            kernel_size=convt_block_kernel_size,
            strides=convt_block_stride,
            dropout_rate=convt_block_dropout_rate,
        )(x)

    x = tf.keras.layers.Conv2DTranspose(3, 3, strides=2, padding="same")(x)

    decoder_outputs = tf.keras.layers.Activation("sigmoid")(x)

    return tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")
