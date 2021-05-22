import tensorflow as tf
from typing import List, Callable


class DAEGraph(tf.keras.Model):
    """Denoising Auto Encoder model"""

    def __init__(
        self,
        encoder_dims: List[int],
        decoder_dims: List[int],
        activation: Callable,
        keep_prob: float = .5
    ):
        super(DAEGraph, self).__init__()

        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims
        self.activation = activation

        self.keep_prob = keep_prob

        self.encoder = self._build_encoder_model()
        self.decoder = self._build_decoder_model()

    def call(self, inputs):
        bootleneck = self.encoder(inputs)
        return self.decoder(bootleneck)

    def _build_encoder_model(self):
        assert self.encoder_dims
        return tf.keras.Sequential(
            [
                tf.keras.Input((self.encoder_dims[0], )),
                tf.keras.layers.Dropout(rate=self.keep_prob)
            ] + [
                tf.keras.layers.Dense(
                    dim,
                    activation=self.activation,
                    name=f'encoder_layer{indx + 1}_{dim}',
                    kernel_initializer=tf.initializers.glorot_uniform(),
                    bias_initializer=tf.initializers.truncated_normal(stddev=0.001),
                    kernel_regularizer="l2"
                ) for indx, dim in enumerate(self.encoder_dims)
            ],
            name='encoder'
        )

    def _build_decoder_model(self):
        assert self.decoder_dims
        return tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    dim,
                    activation=self.activation,
                    name=f'decoder_layer_{indx + 1}_{self.decoder_dims[0]}',
                    kernel_initializer=tf.initializers.glorot_uniform(),
                    bias_initializer=tf.initializers.truncated_normal(stddev=0.001),
                    kernel_regularizer="l2"

                ) for indx, dim in enumerate(self.decoder_dims)
            ],
            name='decoder'
        )
