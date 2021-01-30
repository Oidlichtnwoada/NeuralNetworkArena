import tensorflow as tf

import experiments.models.ct_gru as ct_gru
import experiments.models.ct_rnn as ct_rnn
import experiments.models.differentiable_neural_computer as dnc
import experiments.models.matrix_exponential_unitary_rnn as meurnn
import experiments.models.memory_augmented_transformer as mat
import experiments.models.memory_cell as memory_cell
import experiments.models.neural_circuit_policies as ncp
import experiments.models.ode_lstm as ode_lstm
import experiments.models.recurrent_network_augmented_transformer as rnat
import experiments.models.transformer as transformer
import experiments.models.unitary_ncp as uncp
import experiments.models.unitary_rnn as urnn

MODEL_ARGUMENTS = ['memory_cell', 'memory_augmented_transformer', 'lstm',
                   'differentiable_neural_computer', 'unitary_rnn', 'matrix_exponential_unitary_rnn',
                   'transformer', 'recurrent_network_attention_transformer', 'recurrent_network_augmented_transformer',
                   'gru', 'neural_circuit_policies', 'ct_rnn',
                   'ct_gru', 'ode_lstm', 'unitary_ncp']


def get_concat_inputs(inputs):
    if isinstance(inputs, tuple):
        return tf.concat(inputs, -1)
    else:
        return inputs


def get_concat_input_shape(input_shape):
    if isinstance(input_shape[0], tuple):
        return sum([x[-1] for x in input_shape])
    else:
        return input_shape[-1]


def get_ct_gru_output(output_size, input_tensor):
    return tf.keras.layers.Dense(output_size)(
        tf.keras.layers.RNN(ct_gru.CTGRU(32))(input_tensor))


def get_ct_rnn_output(output_size, input_tensor):
    return tf.keras.layers.Dense(output_size)(
        tf.keras.layers.RNN(ct_rnn.CTRNNCell(128, 'rk4', 3))(input_tensor))


def get_ode_lstm_output(output_size, input_tensor):
    return tf.keras.layers.Dense(output_size)(
        tf.keras.layers.RNN(ode_lstm.ODELSTM(64))(input_tensor))


def get_differentiable_neural_computer_output(output_size, input_tensor):
    return tf.keras.layers.RNN(dnc.DNC(output_size, 64, 16, 8, 2))(input_tensor)


def get_unitary_rnn_output(output_size, input_tensor):
    return tf.keras.layers.Dense(output_size)(
        tf.math.real(tf.keras.layers.RNN(urnn.EUNNCell(128, 16))(input_tensor)))


def get_unitary_ncp_output(output_size, input_tensor):
    return uncp.UnitaryNCP(64, 32, output_size)(input_tensor)


def get_matrix_exponential_unitary_rnn_output(output_size, input_tensor):
    return tf.keras.layers.RNN(meurnn.MatrixExponentialUnitaryRNN(128, output_size))(input_tensor)


def get_lstm_output(output_size, input_tensor):
    return tf.keras.layers.Dense(output_size)(
        tf.keras.layers.LSTM(64)(get_concat_inputs(input_tensor)))


def get_gru_output(output_size, input_tensor):
    return tf.keras.layers.Dense(output_size)(
        tf.keras.layers.GRU(80)(get_concat_inputs(input_tensor)))


def get_transformer_output(output_size, input_tensor):
    return transformer.Transformer(token_amount=1, token_size=output_size, d_model=16, num_heads=2, d_ff=64,
                                   num_layers=2, dropout_rate=0.1, attention=transformer.MultiHeadAttention)(input_tensor)


def get_recurrent_network_attention_transformer_output(output_size, input_tensor):
    return transformer.Transformer(token_amount=1, token_size=output_size, d_model=8, num_heads=1, d_ff=64,
                                   num_layers=1, dropout_rate=0.1, attention=mat.RecurrentNetworkAttention)(input_tensor)


def get_recurrent_network_augmented_transformer_output(output_size, input_tensor):
    return transformer.Transformer(token_amount=1, token_size=output_size, d_model=16, num_heads=2, d_ff=64,
                                   num_layers=1, dropout_rate=0.1, attention=rnat.MultiHeadRecurrentAttention)(input_tensor)


def get_neural_circuit_policies_output(output_size, input_tensor):
    return ncp.NeuralCircuitPolicies(64, output_size)(input_tensor)


def get_memory_augmented_transformer_output(output_size, input_tensor):
    return tf.keras.layers.RNN(mat.MemoryAugmentedTransformerCell(output_size=output_size))(input_tensor)


def get_memory_cell_output(output_size, input_tensor):
    assert output_size == 2
    return tf.keras.layers.RNN(memory_cell.MemoryCell(), return_sequences=True)(input_tensor)


def get_model_output_by_name(model_name, output_size, input_tensor):
    return eval(f'get_{model_name}_output')(output_size, input_tensor if len(input_tensor) > 1 else input_tensor[0])
