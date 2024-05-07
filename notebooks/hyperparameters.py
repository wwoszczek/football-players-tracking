BATCH_SIZE = 128
STEPS_PER_EPOCH = 2106 // BATCH_SIZE
IMAGE_SIZE = (64, 32)
LATENT_DIM = 16

RECONSTRUCTION_LOSS_WEIGHT = 50
KL_LOSS_WEIGHT = 1

ENCODER_MODEL_HYPERPARAMETERS = {
    "image_shape": (64, 32, 3),
    "latent_dim": LATENT_DIM,
    "conv_block_filters": [32, 32, 64],
    "conv_block_kernel_sizes": [3, 3, 3],
    "conv_block_strides": [2, 2, 2],
    "conv_block_dropout_rates": [0.25, 0.25, 0.25],
}

DECODER_MODEL_HYPERPARAMETERS = {
    "latent_dim": LATENT_DIM,
    "dense_layer_units": 2048,
    "reshape_layer_target_shape": (8, 4, 64),
    "convt_block_filters": [64, 32],
    "convt_block_kernel_sizes": [3, 3],
    "convt_block_strides": [2, 2],
    "convt_block_dropout_rates": [0.25, 0.25],
}
