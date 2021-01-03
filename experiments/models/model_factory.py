import tensorflow as tf

import experiments.benchmarks.benchmark as benchmark
import experiments.models.ct_gru as ct_gru
import experiments.models.ct_rnn as ct_rnn
import experiments.models.differentiable_neural_computer as dnc
import experiments.models.enhanced_unitary_rnn as eurnn
import experiments.models.memory_cell as memory_cell
import experiments.models.memory_layer as memory_layer
import experiments.models.neural_circuit_policies as ncp
import experiments.models.ode_lstm as ode_lstm
import experiments.models.recurrent_transformer as recurrent_transformer
import experiments.models.transformer as transformer
import experiments.models.unitary_rnn as urnn


def get_concat_inputs(inputs):
    if isinstance(inputs, tuple):
        return tf.concat(inputs, -1)
    else:
        return inputs


def get_concat_input_shape(input_shape):
    if len(benchmark.get_recursive_shape(input_shape)) > 1:
        return sum([x[-1] for x in input_shape])
    else:
        return input_shape[-1]


def get_model_descriptions():
    return {'memory_cell': True,
            'memory_layer': True,
            'lstm': True,
            'differentiable_neural_computer': True,
            'unitary_rnn': True,
            'enhanced_unitary_rnn': True,
            'transformer': False,
            'memory_layer_transformer': False,
            'recurrent_transformer': False,
            'gru': True,
            'neural_circuit_policies': True,
            'ct_rnn': True,
            'ct_gru': True,
            'ode_lstm': True}


def get_ct_gru_output(output_size, input_tensor, output_per_timestep):
    return tf.keras.layers.Dense(output_size)(
        tf.keras.layers.RNN(ct_gru.CTGRU(76), return_sequences=output_per_timestep)(input_tensor))


def get_ct_rnn_output(output_size, input_tensor, output_per_timestep):
    return tf.keras.layers.Dense(output_size)(
        tf.keras.layers.RNN(ct_rnn.CTRNNCell(312, 'rk4', 3), return_sequences=output_per_timestep)(input_tensor))


def get_ode_lstm_output(output_size, input_tensor, output_per_timestep):
    return tf.keras.layers.Dense(output_size)(
        tf.keras.layers.RNN(ode_lstm.ODELSTM(128), return_sequences=output_per_timestep)(input_tensor))


def get_differentiable_neural_computer_output(output_size, input_tensor, output_per_timestep):
    return tf.keras.layers.RNN(dnc.DNC(output_size, 116, 64, 16, 4), return_sequences=output_per_timestep)(input_tensor)


def get_unitary_rnn_output(output_size, input_tensor, output_per_timestep):
    return tf.keras.layers.Dense(output_size)(
        tf.math.real(tf.keras.layers.RNN(urnn.EUNNCell(8192, 8), return_sequences=output_per_timestep)(input_tensor)))


def get_enhanced_unitary_rnn_output(output_size, input_tensor, output_per_timestep):
    return tf.keras.layers.RNN(eurnn.EnhancedUnitaryRNN(156, output_size), return_sequences=output_per_timestep)(input_tensor)


def get_lstm_output(output_size, input_tensor, output_per_timestep):
    return tf.keras.layers.Dense(output_size)(
        tf.keras.layers.LSTM(160, return_sequences=output_per_timestep)(get_concat_inputs(input_tensor)))


def get_gru_output(output_size, input_tensor, output_per_timestep):
    return tf.keras.layers.Dense(output_size)(
        tf.keras.layers.GRU(100, return_sequences=output_per_timestep)(get_concat_inputs(input_tensor)))


def get_transformer_output(output_size, input_tensor, output_per_timestep):
    return transformer.Transformer(token_amount=1, token_size=output_size, d_model=32, num_heads=2, d_ff=200,
                                   num_layers=2, dropout_rate=0.1, attention=transformer.MultiHeadAttention)(input_tensor)


def get_memory_layer_transformer_output(output_size, input_tensor, output_per_timestep):
    return transformer.Transformer(token_amount=1, token_size=output_size, d_model=32, num_heads=2, d_ff=128,
                                   num_layers=2, dropout_rate=0.1, attention=memory_layer.MemoryLayerAttention)(input_tensor)


def get_recurrent_transformer_output(output_size, input_tensor, output_per_timestep):
    return transformer.Transformer(token_amount=1, token_size=output_size, d_model=24, num_heads=2, d_ff=96,
                                   num_layers=2, dropout_rate=0.1, attention=recurrent_transformer.MultiHeadRecurrentAttention)(input_tensor)


def get_neural_circuit_policies_output(output_size, input_tensor, output_per_timestep):
    return ncp.NeuralCircuitPolicies(
        output_length=output_size, inter_neurons=96, command_neurons=48, motor_neurons=output_size,
        sensory_fanout=8, inter_fanout=8, recurrent_command_synapses=16, motor_fanin=8, output_per_timestep=output_per_timestep)(input_tensor)


def get_memory_layer_output(output_size, input_tensor, output_per_timestep):
    return tf.keras.layers.RNN(memory_layer.MemoryLayerCell(output_size=output_size), return_sequences=output_per_timestep)(input_tensor)


def get_memory_cell_output(output_size, input_tensor, output_per_timestep):
    assert output_size == 2
    return tf.keras.layers.RNN(memory_cell.MemoryCell(), return_sequences=output_per_timestep)(input_tensor)


def get_model_output_by_name(model_name, output_size, input_tensor, output_per_timestep):
    return eval(f'get_{model_name}_output')(output_size, input_tensor if len(input_tensor) > 1 else input_tensor[0], output_per_timestep)
