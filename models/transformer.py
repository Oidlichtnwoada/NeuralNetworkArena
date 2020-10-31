import numpy as np
import tensorflow as tf


def positional_encoding(positions, d_model):
    # compute factors for all dimensions
    positional_encoding_factors = 1 / np.power(1E4, 2 * (np.arange(d_model) // 2) / d_model)
    # multiply each factor with the corresponding position to get the argument for the trigonometric functions
    positional_encoding_matrix = positions.numpy() * positional_encoding_factors
    # apply a sine to the even dimensions
    positional_encoding_matrix[:, :, 0::2] = np.sin(positional_encoding_matrix[:, :, 0::2])
    # apply a cosine to the odd dimensions
    positional_encoding_matrix[:, :, 1::2] = np.cos(positional_encoding_matrix[:, :, 1::2])
    # cast the result and return a tensor
    return tf.cast(positional_encoding_matrix, dtype=tf.float32)


class Encoder(tf.keras.Model):
    def __init__(self, d_model):
        super(Encoder, self).__init__()
        # parameters
        self.d_model = d_model
        # used models or layers
        self.embedding = tf.keras.layers.Dense(self.d_model)

    def call(self, inputs, training=None, mask=None):
        # this function computes the output of the encoder
        signals, times = inputs
        # embed the signal vectors into vectors of size d_model
        embedded_signals = self.embedding(signals)
        # scale with with factor
        embedded_signals *= tf.math.sqrt(tf.cast(self.d_model, dtype=tf.float32))
        # add positional information to the embedded signals using times
        positional_embedded_signals = embedded_signals + positional_encoding(tf.cumsum(times, axis=1), self.d_model)
        return positional_embedded_signals

    def get_config(self):
        pass


class Decoder(tf.keras.Model):
    def __init__(self, token_amount):
        super(Decoder, self).__init__()
        # parameters
        self.token_amount = token_amount
        # used models or layers
        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense_layer = tf.keras.layers.Dense(self.token_amount)

    def call(self, inputs, training=None, mask=None):
        # this function computes the output of the decoder
        flattened_inputs = self.flatten_layer(inputs)
        return self.dense_layer(flattened_inputs)

    def get_config(self):
        pass


class Transformer(tf.keras.Model):
    def __init__(self, token_amount, d_model=512):
        super(Transformer, self).__init__()
        # parameters
        self.token_amount = token_amount
        self.d_model = d_model
        # used models or layers
        self.encoder = Encoder(self.d_model)
        self.decoder = Decoder(self.token_amount)

    def call(self, inputs, training=None, mask=None):
        # build the encoder output
        encoder_output = self.encoder(inputs)
        # build the decoder output
        decoder_output = self.decoder(encoder_output)
        # the output of the transformer is the output of the decoder
        return decoder_output

    def get_config(self):
        pass
