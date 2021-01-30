import tensorflow as tf

import experiments.models.unitary_rnn as urnn


class RecurrentNetworkAttention(tf.keras.layers.Layer):
    def __init__(self, dim, heads):
        super().__init__()
        # save the dimension and the heads of the transformer
        self.dim = dim
        self.heads = heads
        # create the rnn layers
        self.rnn_layers = [tf.keras.layers.RNN(urnn.EUNNCell(self.dim)) for _ in range(self.heads)]
        self.dense_layer = tf.keras.layers.Dense(self.dim)

    def call(self, inputs, **kwargs):
        # split inputs tuple to the arguments
        queries, _, values, _ = inputs
        # bring queries and values to the same shape
        duplicated_queries = tf.repeat(tf.expand_dims(queries, 2), values.shape[1], 2)
        duplicated_values = tf.repeat(tf.expand_dims(values, 1), queries.shape[1], 1)
        # concatenate queries and values together and reshape it to a single batch dimension
        memory_layer_input = tf.reshape(tf.concat([duplicated_queries, duplicated_values], -1), (-1, values.shape[1], 2 * self.dim))
        # accumulate information with memory layer
        accumulated_inputs = tf.concat([tf.math.real(rnn_layer(memory_layer_input)) for rnn_layer in self.rnn_layers], -1)
        # merge outputs of multiple heads to one single representation
        transformed_inputs = self.dense_layer(accumulated_inputs)
        # reshape the output to the right batch size and the right query dimension
        return tf.reshape(transformed_inputs, (-1, queries.shape[1], self.dim)), None
