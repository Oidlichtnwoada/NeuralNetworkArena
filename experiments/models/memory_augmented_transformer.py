import tensorflow as tf

import experiments.models.model_factory as model_factory
import experiments.models.transformer as transformer


@tf.keras.utils.register_keras_serializable()
class MemoryAugmentedTransformerCell(tf.keras.layers.AbstractRNNCell):
    def __init__(self, memory_rows=16, memory_columns=16, output_size=1,
                 embedding_size=64, heads=4, feed_forward_size=256, dropout_rate=0, **kwargs):
        super().__init__(**kwargs)
        self.memory_rows = memory_rows
        self.memory_columns = memory_columns
        self.output_size_value = output_size
        self.output_layer = tf.keras.layers.Dense(output_size)
        self.memory_input_layer = tf.keras.layers.Dense(self.memory_rows + self.memory_columns)
        self.embedding_size = embedding_size
        self.input_embedding = tf.keras.layers.Dense(self.embedding_size)
        self.memory_embedding = tf.keras.layers.Dense(self.embedding_size)
        self.heads = heads
        self.attention = tf.keras.layers.MultiHeadAttention(self.heads, self.embedding_size)
        self.feed_forward_size = feed_forward_size
        self.feed_forward_layer = transformer.feed_forward_network(self.embedding_size, self.feed_forward_size)
        self.layer_normalization = tf.keras.layers.LayerNormalization(epsilon=1E-6)
        self.state_size_value = ((self.memory_rows, self.memory_columns),)
        self.dropout_rate = dropout_rate
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout_rate)
        self.positional_encoding = transformer.positional_encoding(tf.range(1 + self.memory_rows)[tf.newaxis, ..., tf.newaxis], self.embedding_size)

    @property
    def state_size(self):
        return self.state_size_value

    @property
    def output_size(self):
        return self.output_size_value

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return tf.fill((batch_size, self.memory_rows, self.memory_columns), 1E-6)

    def call(self, inputs, states):
        inputs = model_factory.get_concat_inputs(inputs)
        memory_state = states[0]
        embedded_memory_contents = self.memory_embedding(memory_state)
        embedded_inputs = self.input_embedding(tf.expand_dims(inputs, -2))
        augmented_inputs = tf.concat((embedded_inputs, embedded_memory_contents), -2) + self.positional_encoding
        augmented_inputs = self.dropout_layer(augmented_inputs)
        attention_output = self.dropout_layer(self.attention(augmented_inputs, augmented_inputs)) + augmented_inputs
        normed_attention_output = self.layer_normalization(attention_output)
        feed_forward_output = self.dropout_layer(self.feed_forward_layer(normed_attention_output)) + normed_attention_output
        normed_feed_forward_output = self.layer_normalization(feed_forward_output)
        transformer_output_flattened = tf.reshape(normed_feed_forward_output, (-1, (1 + self.memory_rows) * self.embedding_size))
        memory_layer_outputs = self.output_layer(transformer_output_flattened)
        memory_inputs = self.memory_input_layer(transformer_output_flattened)
        control_signals = tf.expand_dims(memory_inputs[:, :self.memory_rows], -1)
        data_signals = tf.expand_dims(memory_inputs[:, -self.memory_columns:], -2)
        memory_state = tf.sigmoid(-control_signals) * memory_state + tf.sigmoid(control_signals) * data_signals
        return memory_layer_outputs, (memory_state,)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'memory_rows': self.memory_rows,
            'memory_columns': self.memory_columns,
            'output_size': self.output_size,
            'embedding_size': self.embedding_size,
            'heads': self.heads,
            'feed_forward_size': self.feed_forward_size,
            'dropout_rate': self.dropout_rate
        })
        return config


class MemoryLayerAttention(tf.keras.layers.Layer):
    def __init__(self, dim, heads):
        super().__init__()
        # save the dimension and the heads of the transformer
        self.dim = dim
        self.heads = heads
        # create a memory layer
        self.memory_layer = tf.keras.layers.RNN(MemoryAugmentedTransformerCell(heads=self.heads, output_size=self.dim, embedding_size=24, feed_forward_size=64))

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
