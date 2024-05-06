import tensorflow as tf
from .blocks import DenseBlock, ConvBlock


def build_multilabel_classifier(
    encoder_model: tf.keras.Model,
    dense_block_units: list[int],
    dense_block_dropout_rates: list[float],
    n_unique_features: list[int],
    feature_names: list[str],
) -> tf.keras.Model:
    for i in range(len(encoder_model.layers)):
        encoder_model.layers[i].trainable = False

    inp = tf.keras.layers.Input((256, 256, 3))
    x = encoder_model(inp)
    x = tf.keras.layers.Concatenate()([x[0], x[1], x[2]])

    for dense_block_units, dense_block_dropout_rate in zip(dense_block_units, dense_block_dropout_rates):
        x = DenseBlock(units=dense_block_units, dropout_rate=dense_block_dropout_rate)(x)

    outputs = []
    for n, feature_name in zip(n_unique_features, feature_names):
        outputs.append(tf.keras.layers.Dense(n, activation="sigmoid", name=feature_name)(x))

    return tf.keras.Model(inp, outputs, name="classifier")


def build_single_label_classifier(
    image_cropping: tuple[tuple[int, int], tuple[int, int]],
    conv_block_filters: list[int],
    conv_block_kernel_sizes: list[int],
    conv_block_strides: list[int],
    conv_block_dropout_rates: list[float],
    dense_block_units: list[int],
    dense_block_dropout_rates: list[float],
    n_unique_features: int,
) -> tf.keras.Model:
    inp = tf.keras.layers.Input((256, 256, 3))
    x = tf.keras.layers.Cropping2D(cropping=image_cropping)(inp)

    for conv_block_filter, conv_block_kernel_size, conv_block_stride, conv_block_dropout_rate in zip(
        conv_block_filters, conv_block_kernel_sizes, conv_block_strides, conv_block_dropout_rates
    ):
        x = ConvBlock(
            filters=conv_block_filter,
            kernel_size=conv_block_kernel_size,
            strides=conv_block_stride,
            dropout_rate=conv_block_dropout_rate,
        )(x)

    x = tf.keras.layers.Flatten()(x)

    for dense_block_unit, dense_block_dropout_rate in zip(dense_block_units, dense_block_dropout_rates):
        x = DenseBlock(units=dense_block_unit, dropout_rate=dense_block_dropout_rate)(x)

    x = tf.keras.layers.Dense(n_unique_features, activation="sigmoid")(x)

    return tf.keras.Model(inp, x)
