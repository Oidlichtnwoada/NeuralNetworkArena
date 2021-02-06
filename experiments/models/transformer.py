import tensorflow as tf

import experiments.models.recurrent_network_attention as rna
import experiments.models.recurrent_network_augmented_transformer as rnat


def compute_padding_mask(signals):
    # mask the input away if all vector entries are zero
    padding_mask = tf.reduce_max(tf.cast(signals != 0, dtype=tf.float32), axis=2)
    # adjust dimension to enable batch operation during dot product attention
    return padding_mask[:, tf.newaxis, :]


def compute_look_ahead_mask(signals):
    # make sure that no attention to future positions is possible
    look_ahead_mask = tf.linalg.band_part(tf.ones((signals.shape[1], signals.shape[1])), -1, 0)
    # adjust dimension to enable batch operation during dot product attention
    return look_ahead_mask[tf.newaxis, ...]


def positional_encoding(positions, d_model):
    # compute factors for all dimensions
    positional_encoding_factors = 1 / tf.pow(1E4, tf.cast(2 * (tf.range(d_model) // 2) / d_model, dtype=tf.float32))
    # multiply each factor with the corresponding position to get the argument for the trigonometric functions
    positional_encoding_matrix = tf.cast(positions, dtype=tf.float32) * positional_encoding_factors
    # apply a sine to the even dimensions
    pem_sine = tf.sin(positional_encoding_matrix[:, :, 0::2])
    # apply a cosine to the odd dimensions
    pem_cosine = tf.cos(positional_encoding_matrix[:, :, 1::2])
    # cast the result and return a tensor
    return tf.reshape(tf.concat([pem_sine[..., tf.newaxis], pem_cosine[..., tf.newaxis]], axis=-1), (-1, positional_encoding_matrix.shape[1], positional_encoding_matrix.shape[2]))


def feed_forward_network(d_model, d_ff):
    # return the feed forward network structure used in the transformer layers
    return tf.keras.Sequential([
        # a dense layer with relu activation
        tf.keras.layers.Dense(d_ff, activation='relu'),
        # a dense layer without an activation function
        tf.keras.layers.Dense(d_model)
    ])


def split_heads(qkv, num_heads, d_qkv):
    # split queries, key or values into num_heads - permutation necessary to compute right dot product
    return tf.transpose(tf.reshape(qkv, (-1, qkv.shape[1], num_heads, d_qkv)), perm=[0, 2, 1, 3])


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        # parameters
        self.d_model = d_model
        self.num_heads = num_heads
        # used layers
        self.query_generator_network = tf.keras.layers.Dense(self.num_heads * self.d_model)
        self.key_generator_network = tf.keras.layers.Dense(self.num_heads * self.d_model)
        self.value_generator_network = tf.keras.layers.Dense(self.num_heads * self.d_model)
        self.mha_output_generator_network = tf.keras.layers.Dense(self.d_model)

    def call(self, inputs, **kwargs):
        # split inputs tuple to the arguments
        query_gen_input, key_gen_input, value_gen_input, mask = inputs
        # bring mask to format where 1 denotes no attention and insert head dimension
        if mask is not None:
            mask = tf.expand_dims(1 - mask, 1)
        # generate queries, keys and values
        queries = self.query_generator_network(query_gen_input)
        keys = self.key_generator_network(key_gen_input)
        values = self.value_generator_network(value_gen_input)
        # split queries, keys and values to the right amount of heads
        queries_heads = split_heads(queries, self.num_heads, self.d_model)
        keys_heads = split_heads(keys, self.num_heads, self.d_model)
        values_heads = split_heads(values, self.num_heads, self.d_model)
        # compute the attention logits from each query to each key
        attention_logits = tf.matmul(queries_heads, keys_heads, transpose_b=True)
        # scale the attention logits
        scaled_attention_logits = attention_logits / tf.math.sqrt(tf.cast(self.d_model, dtype=tf.float32))
        # set attention logits to very small value for input positions in mask (if present)
        if mask is not None:
            scaled_attention_logits -= tf.where(mask == 1, tf.ones_like(mask) * float('inf'), mask)
        # compute the attention weight to each value per query
        attention_weights = tf.nn.softmax(scaled_attention_logits)
        # compute dot product attention
        dpa = tf.matmul(attention_weights, values_heads)
        # transpose dpa matrix such that the heads dimension is behind input dimension
        reshaped_dpa = tf.transpose(dpa, perm=[0, 2, 1, 3])
        # merge heads to single value dimension
        concatenated_dpa = tf.reshape(reshaped_dpa, (-1, reshaped_dpa.shape[1], self.num_heads * self.d_model))
        # transform concatenated dpa to vectors of size d_model
        return self.mha_output_generator_network(concatenated_dpa), attention_weights


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate, attention):
        super().__init__()
        # parameters
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.attention = attention
        # used layers
        self.att = self.attention(self.d_model, self.num_heads)
        self.ffn = feed_forward_network(self.d_model, self.d_ff)
        self.att_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.att_dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.ffn_dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, inputs, **kwargs):
        # split inputs tuple to the arguments
        signals, encoder_zero_input_mask = inputs
        # compute self attention output values
        att_output, attention_weights = self.att((signals, signals, signals, encoder_zero_input_mask))
        # use a dropout layer to prevent overfitting
        att_output = self.att_dropout(att_output)
        # normalize att output with residual connection
        att_layer_norm_output = self.att_layer_norm(signals + att_output)
        # compute feed forward network output values
        ffn_output = self.ffn(att_layer_norm_output)
        # use a dropout layer to prevent overfitting
        ffn_output = self.ffn_dropout(ffn_output)
        # normalize ffn output with residual connection
        ffn_layer_norm_output = self.ffn_layer_norm(att_layer_norm_output + ffn_output)
        # the output of the second normalization layer is the output of the encoder layer
        return ffn_layer_norm_output


class Encoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, num_layers, mask_zero_inputs, dropout_rate, attention):
        super().__init__()
        # parameters
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.mask_zero_inputs = mask_zero_inputs
        self.dropout_rate = dropout_rate
        self.attention = attention
        # used layers
        self.embedding = tf.keras.layers.Dense(self.d_model)
        self.encoder_layers = [EncoderLayer(self.d_model, self.num_heads, self.d_ff, self.dropout_rate, self.attention) for _ in range(self.num_layers)]
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, inputs, **kwargs):
        # this function computes the output of the encoder
        if isinstance(inputs, tuple):
            encoder_input, times = inputs
            positional_encoding_matrix = positional_encoding(tf.cumsum(times, axis=1), self.d_model)
        else:
            encoder_input = inputs
            positional_encoding_matrix = positional_encoding(tf.range(encoder_input.shape[1])[tf.newaxis, :, tf.newaxis], self.d_model)
        # compute mask to not attend to zero inputs if enabled
        if self.mask_zero_inputs:
            encoder_zero_input_mask = compute_padding_mask(encoder_input)
        else:
            encoder_zero_input_mask = None
        # embed the signal vectors into vectors of size d_model
        embedded_signals = self.embedding(encoder_input)
        # scale with with factor
        embedded_signals *= tf.math.sqrt(tf.cast(self.d_model, dtype=tf.float32))
        # add positional information to the embedded signals using times
        positional_embedded_signals = embedded_signals + positional_encoding_matrix
        # use a dropout layer to prevent overfitting
        positional_embedded_signals = self.dropout_layer(positional_embedded_signals)
        # create variable that is updated by each encoder layer
        encoder_layer_inout = positional_embedded_signals
        for i in range(self.num_layers):
            # compute output of each encoder layer
            encoder_layer_inout = self.encoder_layers[i]((encoder_layer_inout, encoder_zero_input_mask))
        # the output of the last encoder layer is the output of the encoder
        return encoder_layer_inout, encoder_zero_input_mask


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate, attention):
        super().__init__()
        # parameters
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.attention = attention
        # used layers
        self.self_att = self.attention(self.d_model, self.num_heads)
        self.enc_dec_att = self.attention(self.d_model, self.num_heads)
        self.ffn = feed_forward_network(self.d_model, self.d_ff)
        self.self_att_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.enc_dec_att_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.self_att_dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.enc_dec_att_dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.ffn_dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, inputs, **kwargs):
        # split inputs tuple to the arguments
        signals, encoder_output, decoder_zero_input_mask, look_ahead_mask = inputs
        # compute self attention output values
        self_att_output, attention_weights = self.self_att((signals, signals, signals, look_ahead_mask))
        # use a dropout layer to prevent overfitting
        self_att_output = self.self_att_dropout(self_att_output)
        # normalize self att output with residual connection
        self_att_layer_norm_output = self.self_att_layer_norm(signals + self_att_output)
        # compute encoder decoder att output values
        enc_dec_att_output, attention_weights = self.enc_dec_att((self_att_layer_norm_output, encoder_output, encoder_output, decoder_zero_input_mask))
        # use a dropout layer to prevent overfitting
        enc_dec_att_output = self.enc_dec_att_dropout(enc_dec_att_output)
        # normalize encoder decoder att output with residual connection
        enc_dec_att_layer_norm_output = self.enc_dec_att_layer_norm(self_att_layer_norm_output + enc_dec_att_output)
        # compute feed forward network output values
        ffn_output = self.ffn(enc_dec_att_layer_norm_output)
        # use a dropout layer to prevent overfitting
        ffn_output = self.ffn_dropout(ffn_output)
        # normalize ffn output with residual connection
        ffn_layer_norm_output = self.ffn_layer_norm(enc_dec_att_layer_norm_output + ffn_output)
        # the output of the third normalization layer is the output of the decoder layer
        return ffn_layer_norm_output


class Decoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, num_layers, token_amount, token_size, mask_zero_inputs, dropout_rate, attention):
        super().__init__()
        # parameters
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.token_amount = token_amount
        self.token_size = token_size
        self.mask_zero_inputs = mask_zero_inputs
        self.dropout_rate = dropout_rate
        self.attention = attention
        # used layers
        self.embedding = tf.keras.layers.Dense(self.d_model)
        self.token_output_layer = tf.keras.layers.Dense(self.token_size)
        self.decoder_layers = [DecoderLayer(self.d_model, self.num_heads, self.d_ff, self.dropout_rate, self.attention) for _ in range(self.num_layers)]
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, inputs, **kwargs):
        # split inputs tuple to the arguments
        encoder_output, decoder_zero_input_mask = inputs
        # create a start token
        tokens = tf.repeat(tf.ones_like(encoder_output)[:, :1, :1], self.token_size, axis=-1)
        # create the right amount of tokens
        for _ in range(self.token_amount):
            # create a look ahead mask such that tokens can only attend to previous positions
            look_ahead_mask = compute_look_ahead_mask(tokens)
            # embed the current tokens
            embedded_tokens = self.embedding(tokens)
            # scale with with factor
            embedded_tokens *= tf.math.sqrt(tf.cast(self.d_model, dtype=tf.float32))
            # build the position matrix
            positions = tf.range(embedded_tokens.shape[1])[tf.newaxis, :, tf.newaxis]
            # add positional information to the embedded tokens
            positional_embedded_tokens = embedded_tokens + positional_encoding(positions, self.d_model)
            # use a dropout layer to prevent overfitting
            positional_embedded_tokens = self.dropout_layer(positional_embedded_tokens)
            # create variable that is updated by each decoder layer
            decoder_layer_inout = positional_embedded_tokens
            for i in range(self.num_layers):
                # compute output of each decoder layer
                decoder_layer_inout = self.decoder_layers[i]((decoder_layer_inout, encoder_output, decoder_zero_input_mask, look_ahead_mask))
            # the output of the last decoder layer must be fed to the output dense layer
            # only the output token corresponding to the last input token is used
            next_token = self.token_output_layer(decoder_layer_inout[:, -1:, :])
            # add the new token to the token matrix
            tokens = tf.concat([tokens, next_token], axis=1)
        # return all produced tokens except the start token
        return tokens[:, 1:, :]


@tf.keras.utils.register_keras_serializable()
class Transformer(tf.keras.layers.Layer):
    def __init__(self, token_amount, token_size, d_model, num_heads, d_ff, num_layers, dropout_rate, attention, flatten_output=True, mask_zero_inputs=False, **kwargs):
        super().__init__(**kwargs)
        # parameters
        self.token_amount = token_amount
        self.token_size = token_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.flatten_output = flatten_output
        self.mask_zero_inputs = mask_zero_inputs
        self.dropout_rate = dropout_rate
        self.attention_type = attention
        if self.attention_type == 'mha':
            self.attention = MultiHeadAttention
        elif self.attention_type == 'rna':
            self.attention = rna.RecurrentNetworkAttention
        elif self.attention_type == 'rnat':
            self.attention = rnat.MultiHeadRecurrentAttention
        else:
            raise NotImplementedError
        # used layers
        self.encoder = Encoder(self.d_model, self.num_heads, self.d_ff, self.num_layers, self.mask_zero_inputs, self.dropout_rate, self.attention)
        self.decoder = Decoder(self.d_model, self.num_heads, self.d_ff, self.num_layers, self.token_amount, self.token_size, self.mask_zero_inputs, self.dropout_rate, self.attention)
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs, **kwargs):
        # build the encoder output
        encoder_output = self.encoder(inputs)
        # build the decoder output
        decoder_output = self.decoder(encoder_output)
        # the output of the transformer is the (flattened) output of the decoder
        if self.flatten_output:
            return self.flatten(decoder_output)
        else:
            return decoder_output

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'token_amount': self.token_amount,
            'token_size': self.token_size,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'd_ff': self.d_ff,
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate,
            'attention': self.attention_type,
            'flatten_output': self.flatten_output,
            'mask_zero_inputs': self.mask_zero_inputs
        })
        return config
