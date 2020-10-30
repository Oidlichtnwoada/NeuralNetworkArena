import tensorflow as tf


def positional_embedding(positions, d_model):
    pass


def encoder_output(inputs, d_model):
    # this function computes the output of the encoder
    signals, times = inputs
    # embed the signal vectors into vectors of size d_model
    embedded_signals = tf.keras.layers.Dense(d_model)(signals)
    # add positional information to the embedded signals using times
    # positional_embedded_signals = embedded_signals + positional_embedding(tf.cumsum(times, axis=-1), d_model)
    return embedded_signals


def decoder_output(inputs, token_amount):
    # this function computes the output of the decoder
    flattened_inputs = tf.keras.layers.Flatten()(inputs)
    return tf.keras.layers.Dense(token_amount)(flattened_inputs)


def transformer(inputs, token_amount, d_model=512):
    # build the encoder output
    enc_output = encoder_output(inputs, d_model)
    # build the decoder output
    dec_output = decoder_output(enc_output, token_amount)
    # the output of the transformer is the output of the decoder
    return dec_output
