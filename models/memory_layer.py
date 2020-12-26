import tensorflow as tf

from models.transformer import positional_encoding, feed_forward_network


class MemoryLayerCell(tf.keras.layers.AbstractRNNCell):
    def __init__(self, memory_rows=32, memory_columns=8, output_size=1,
                 embedding_size=32, heads=4, feed_forward_size=128):
        super().__init__()
        self.memory_rows = memory_rows
        self.memory_columns = memory_columns
        self.output_size_value = output_size
        self.output_layer = tf.keras.layers.Dense(output_size)
        self.memory_input_layer = tf.keras.layers.Dense(1 + self.memory_columns)
        self.embedding_size = embedding_size
        self.input_embedding = tf.keras.layers.Dense(self.embedding_size)
        self.memory_embedding = tf.keras.layers.Dense(self.embedding_size)
        self.heads = heads
        self.attention = tf.keras.layers.MultiHeadAttention(self.heads, self.embedding_size)
        self.feed_forward_size = feed_forward_size
        self.feed_forward_layer = feed_forward_network(self.embedding_size, self.feed_forward_size)
        self.layer_normalization = tf.keras.layers.LayerNormalization(epsilon=1E-6)
        self.state_size_value = ((self.memory_rows, self.memory_columns),)
        self.positional_encoding = positional_encoding(tf.range(1 + self.memory_rows)[tf.newaxis, ..., tf.newaxis], self.embedding_size)

    @property
    def state_size(self):
        return self.state_size_value

    @property
    def output_size(self):
        return self.output_size_value

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return tf.fill((batch_size, self.memory_rows, self.memory_columns), 1E-6)

    def call(self, inputs, states):
        memory_state = states[0]
        embedded_memory_contents = self.memory_embedding(memory_state)
        embedded_inputs = self.input_embedding(tf.expand_dims(inputs, -2))
        augmented_inputs = tf.concat((embedded_inputs, embedded_memory_contents), -2) + self.positional_encoding
        attention_output = self.attention(augmented_inputs, augmented_inputs) + augmented_inputs
        normed_attention_output = self.layer_normalization(attention_output)
        feed_forward_output = self.feed_forward_layer(normed_attention_output) + normed_attention_output
        normed_feed_forward_output = self.layer_normalization(feed_forward_output)
        memory_layer_outputs = self.output_layer(normed_feed_forward_output[:, 0, :])
        memory_inputs = self.memory_input_layer(normed_feed_forward_output[:, 1:, :])
        control_signals = tf.sigmoid(memory_inputs[..., :1])
        data_signals = memory_inputs[..., 1:]
        memory_state = tf.sigmoid(-control_signals) * memory_state + tf.sigmoid(control_signals) * data_signals
        return memory_layer_outputs, (memory_state,)


class MemoryLayerAttention(tf.keras.layers.Layer):
    def __init__(self, dim, heads):
        super().__init__()
        # save the dimension and the heads of the transformer
        self.dim = dim
        self.heads = heads
        # create a memory layer out of heads times dim memory cells and an output size of dim
        self.memory_layer = tf.keras.layers.RNN(MemoryLayerCell(2 * self.heads * self.dim, self.dim))

    def call(self, inputs, **kwargs):
        # split inputs tuple to the arguments
        queries, _, values, _ = inputs
        # bring queries and values to the same shape
        duplicated_queries = tf.repeat(tf.expand_dims(queries, 2), values.shape[1], 2)
        duplicated_values = tf.repeat(tf.expand_dims(values, 1), queries.shape[1], 1)
        # concatenate queries and values together and reshape it to a single batch dimension
        memory_layer_input = tf.reshape(tf.concat([duplicated_queries, duplicated_values], -1), (-1, values.shape[1], 2 * self.dim))
        # accumulate information with memory layer
        accumulated_inputs = self.memory_layer(memory_layer_input)
        # reshape the output to the right batch size and the right query dimension
        return tf.reshape(accumulated_inputs, (-1, queries.shape[1], self.dim)), None
