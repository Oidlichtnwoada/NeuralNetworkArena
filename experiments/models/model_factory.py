import tensorflow as tf

import experiments.models.differentiable_neural_computer as dnc
import experiments.models.enhanced_unitary_rnn as eurnn
import experiments.models.memory_layer as memory_layer
import experiments.models.neural_circuit_policies as ncp
import experiments.models.recurrent_transformer as recurrent_transformer
import experiments.models.transformer as transformer
import experiments.models.unitary_rnn as urnn


def get_model_descriptions():
    return {'memory_layer': True,
            'lstm': True,
            'differentiable_neural_computer': True,
            'unitary_rnn': True,
            'enhanced_unitary_rnn': True,
            'transformer': False,
            'memory_layer_transformer': False,
            'recurrent_transformer': False,
            'neural_circuit_policies': True}


def get_differentiable_neural_computer_output(output_size, input_tensor):
    return tf.keras.layers.RNN(dnc.DNC(output_size, 100, 64, 16, 4), return_sequences=True)(input_tensor)


def get_unitary_rnn_output(output_size, input_tensor):
    return tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(output_size))(
        tf.math.real(tf.keras.layers.RNN(urnn.EUNNCell(128, 4), return_sequences=True)(input_tensor)))


def get_enhanced_unitary_rnn_output(output_size, input_tensor):
    return tf.keras.layers.RNN(eurnn.EnhancedUnitaryRNN(128, output_size), return_sequences=True)(input_tensor)


def get_lstm_output(output_size, input_tensor):
    return tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(output_size))(
        tf.keras.layers.LSTM(40, return_sequences=True)(input_tensor))


def get_transformer_output(output_size, input_tensor):
    return transformer.Transformer(token_amount=1, token_size=output_size, d_model=64, num_heads=4, d_ff=128,
                                   num_layers=4, dropout_rate=0.1, attention=transformer.MultiHeadAttention)(input_tensor)


def get_memory_layer_transformer_output(output_size, input_tensor):
    return transformer.Transformer(token_amount=1, token_size=output_size, d_model=64, num_heads=4, d_ff=128,
                                   num_layers=4, dropout_rate=0.1, attention=memory_layer.MemoryLayerAttention)(input_tensor)


def get_recurrent_transformer_output(output_size, input_tensor):
    return transformer.Transformer(token_amount=1, token_size=output_size, d_model=64, num_heads=4, d_ff=128,
                                   num_layers=4, dropout_rate=0.1, attention=recurrent_transformer.MultiHeadRecurrentAttention)(input_tensor)


def get_neural_circuit_policies_output(output_size, input_tensor):
    return ncp.NeuralCircuitPolicies(
        output_length=output_size, inter_neurons=32, command_neurons=16, motor_neurons=output_size,
        sensory_fanout=4, inter_fanout=4, recurrent_command_synapses=8, motor_fanin=6)(input_tensor)


def get_memory_layer_output(output_size, input_tensor):
    return tf.keras.layers.RNN(memory_layer.MemoryLayerCell(output_size=output_size), return_sequences=True)(input_tensor)


def get_model_output_by_name(model_name, output_size, input_tensor):
    return eval(f'get_{model_name}_output')(output_size, input_tensor if len(input_tensor) > 1 else input_tensor[0])
