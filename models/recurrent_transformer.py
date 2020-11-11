import tensorflow as tf

from models.transformer import split_heads
from supplements.eunn import EUNNCell


def recurrent_dot_product_attention(queries, keys, values, d_qkv, recurrent_network_layers, mask):
    # compute the attention logits from each query to each key
    attention_logits = tf.matmul(queries, keys, transpose_b=True)
    # scale the attention logits
    scaled_attention_logits = attention_logits / tf.math.sqrt(tf.cast(d_qkv, dtype=tf.float32))
    # set attention logits to very small value for input positions in mask (if present)
    if mask is not None:
        scaled_attention_logits -= tf.where(mask == 1, tf.ones_like(mask) * float('inf'), mask)
    # compute the attention weight to each value per query
    attention_weights = tf.nn.softmax(scaled_attention_logits)
    # duplicate all value vectors for each input and weight them accordingly
    weighted_value_vectors = tf.expand_dims(attention_weights, axis=-1) * tf.repeat(tf.expand_dims(values, axis=2), values.shape[2], axis=2)
    # this variable holds the concatenated rnn output at end
    concatenated_rnn_output = tf.ones_like(values)[:, :0, :, :]
    for head_index, layer in enumerate(recurrent_network_layers):
        # aggregate all weighted value vectors for each input via an rnn for each head instead of a simple summation
        shape = (-1,) + (weighted_value_vectors.shape[2],) + (weighted_value_vectors.shape[4],)
        rnn_input = tf.reshape(weighted_value_vectors[:, head_index, :, :, :], shape)
        rnn_output = tf.reshape(recurrent_network_layers[head_index](rnn_input), shape)
        # concatenate the real part of the output for this head to the running variable
        concatenated_rnn_output = tf.concat([concatenated_rnn_output, tf.expand_dims(tf.math.real(rnn_output), axis=1)], axis=1)
    # return the concatenated result of all heads
    return concatenated_rnn_output, attention_weights


class MultiHeadRecurrentAttentionEUNN(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadRecurrentAttentionEUNN, self).__init__()
        # parameters
        self.d_model = d_model
        self.num_heads = num_heads
        # set size of queries, keys and values to d_model / num_heads
        assert self.d_model % self.num_heads == 0
        self.d_qkv = self.d_model // self.num_heads
        # used layers
        self.query_generator_network = tf.keras.layers.Dense(self.d_model)
        self.key_generator_network = tf.keras.layers.Dense(self.d_model)
        self.value_generator_network = tf.keras.layers.Dense(self.d_model)
        self.mhra_output_generator_network = tf.keras.layers.Dense(self.d_model)
        self.recurrent_network_layers = [tf.keras.layers.RNN(EUNNCell(self.d_qkv)) for _ in range(self.num_heads)]

    def call(self, inputs, **kwargs):
        # split inputs tuple to the arguments
        query_gen_input, key_gen_input, value_gen_input, mask = inputs
        # generate queries, keys and values
        queries = self.query_generator_network(query_gen_input)
        keys = self.key_generator_network(key_gen_input)
        values = self.value_generator_network(value_gen_input)
        # split queries, keys and values to the right amount of heads
        queries_heads = split_heads(queries, self.num_heads, self.d_qkv)
        keys_heads = split_heads(keys, self.num_heads, self.d_qkv)
        value_heads = split_heads(values, self.num_heads, self.d_qkv)
        # compute the recurrent dot product attention
        rdpa, attention_weights = recurrent_dot_product_attention(queries_heads, keys_heads, value_heads, self.d_qkv, self.recurrent_network_layers, mask)
        # transpose rdpa matrix such that the heads dimension is behind input dimension
        reshaped_rdpa = tf.transpose(rdpa, perm=[0, 2, 1, 3])
        # merge heads to single value dimension
        concatenated_rdpa = tf.reshape(reshaped_rdpa, (-1,) + (reshaped_rdpa.shape[1],) + (self.d_model,))
        # transform concatenated dpa to vectors of size d_model
        return self.mhra_output_generator_network(concatenated_rdpa), attention_weights
