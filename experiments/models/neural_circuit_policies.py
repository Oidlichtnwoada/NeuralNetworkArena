import kerasncp as ncp
import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class NeuralCircuitPolicies(tf.keras.Model):
    def __init__(self, output_length, inter_neurons, command_neurons, motor_neurons, sensory_fanout, inter_fanout,
                 recurrent_command_synapses, motor_fanin, output_per_timestep, **kwargs):
        super().__init__(**kwargs)
        # parameters
        self.output_length = output_length
        self.inter_neurons = inter_neurons
        self.command_neurons = command_neurons
        self.motor_neurons = motor_neurons
        self.sensory_fanout = sensory_fanout
        self.inter_fanout = inter_fanout
        self.recurrent_command_synapses = recurrent_command_synapses
        self.motor_fanin = motor_fanin
        self.output_per_timestep = output_per_timestep
        # used layers
        self.rnn = tf.keras.layers.RNN(
            ncp.LTCCell(
                ncp.wirings.NCP(self.inter_neurons, self.command_neurons, self.motor_neurons, self.sensory_fanout, self.inter_fanout, self.recurrent_command_synapses, self.motor_fanin)),
            return_sequences=self.output_per_timestep)
        self.dense_layer = tf.keras.layers.Dense(self.output_length)

    def call(self, inputs, training=None, mask=None):
        rnn_output = self.rnn(inputs)
        return self.dense_layer(rnn_output)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_length': self.output_length,
            'inter_neurons': self.inter_neurons,
            'command_neurons': self.command_neurons,
            'motor_neurons': self.motor_neurons,
            'sensory_fanout': self.sensory_fanout,
            'inter_fanout': self.inter_fanout,
            'recurrent_command_synapses': self.recurrent_command_synapses,
            'motor_fanin': self.motor_fanin,
            'output_per_timestep': self.output_per_timestep
        })
        return config
