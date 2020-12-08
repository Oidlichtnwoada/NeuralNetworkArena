import tensorflow as tf


def positional_encoding(positions, embedding_size):
    positional_encoding_factors = 1 / tf.pow(1E4, tf.cast(2 * (tf.range(embedding_size) // 2) / embedding_size, dtype=tf.float32))
    positional_encoding_matrix = tf.cast(tf.range(positions)[tf.newaxis, :, tf.newaxis], dtype=tf.float32) * positional_encoding_factors
    pem_sine = tf.sin(positional_encoding_matrix[:, :, 0::2])
    pem_cosine = tf.cos(positional_encoding_matrix[:, :, 1::2])
    return tf.reshape(tf.concat([pem_sine[..., tf.newaxis], pem_cosine[..., tf.newaxis]], axis=-1), (-1, positions, embedding_size))


class RecurrentMemoryCell(tf.keras.layers.Layer):
    def __init__(self, memory_rows, memory_columns, output_size, embedding_size, controller_heads, feed_forward_size):
        super().__init__()
        self.memory_rows = memory_rows
        self.memory_columns = memory_columns
        self.state_size = self.memory_rows * self.memory_columns
        self.output_size = output_size
        self.output_layer = tf.keras.layers.Dense(self.output_size)
        self.embedding_size = embedding_size
        self.inputs_embedding = tf.keras.layers.Dense(self.embedding_size)
        self.memory_embedding = tf.keras.layers.Dense(self.embedding_size)
        self.controller_heads = controller_heads
        self.controller = tf.keras.layers.MultiHeadAttention(self.controller_heads, self.embedding_size, dropout=1E-1)
        self.layer_normalization = tf.keras.layers.LayerNormalization(epsilon=1E-6)
        self.feed_forward_size = feed_forward_size
        self.feed_forward = tf.keras.Sequential([tf.keras.layers.Dense(self.feed_forward_size, activation='relu'), tf.keras.layers.Dense(self.embedding_size)])

    def call(self, inputs, states):
        expanded_inputs = tf.expand_dims(inputs, axis=1)
        memory = tf.reshape(states[0], (-1, self.memory_rows, self.memory_columns))
        embedded_inputs = self.inputs_embedding(expanded_inputs)
        embedded_memory = self.memory_embedding(memory)
        embedded_controller_input = tf.concat((embedded_inputs, embedded_memory), axis=1)
        positional_embedded_controller_input = embedded_controller_input + positional_encoding(embedded_controller_input.shape[1], self.embedding_size)
        partial_controller_output = self.controller(positional_embedded_controller_input, positional_embedded_controller_input)
        controller_output = self.layer_normalization(positional_embedded_controller_input + partial_controller_output)
        transformed_controller_output = self.layer_normalization(controller_output + self.feed_forward(controller_output))
        output = self.output_layer(transformed_controller_output[:, 0, :])
        memory += transformed_controller_output[:, 1:, :]
        return output, tf.reshape(memory, (-1, self.state_size))
