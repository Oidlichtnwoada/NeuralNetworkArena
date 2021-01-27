import kerasncp as ncp
import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class NeuralCircuitPolicies(tf.keras.layers.Layer):
    def __init__(self, units, output_size, inter_neuron_percentage=0.6, sensory_fanout=4, inter_fanout=4,
                 recurrent_command_synapses=4, motor_fanin=6, return_sequences=False, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.inter_neuron_percentage = inter_neuron_percentage
        self.inter_neurons = int(0.6 * self.units)
        self.command_neurons = self.units - self.inter_neurons
        self.motor_neurons = output_size
        self.sensory_fanout = sensory_fanout
        self.inter_fanout = inter_fanout
        self.recurrent_command_synapses = recurrent_command_synapses
        self.motor_fanin = motor_fanin
        self.return_sequences = return_sequences
        self.rnn = tf.keras.layers.RNN(
            ncp.LTCCell(
                ncp.wirings.NCP(self.inter_neurons, self.command_neurons, self.motor_neurons, self.sensory_fanout, self.inter_fanout, self.recurrent_command_synapses, self.motor_fanin)),
            return_sequences=self.return_sequences)

    def call(self, inputs, **kwargs):
        return self.rnn(inputs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'output_size': self.motor_neurons,
            'inter_neuron_percentage': self.inter_neuron_percentage,
            'sensory_fanout': self.sensory_fanout,
            'inter_fanout': self.inter_fanout,
            'recurrent_command_synapses': self.recurrent_command_synapses,
            'motor_fanin': self.motor_fanin,
            'return_sequences': self.return_sequences
        })
        return config
