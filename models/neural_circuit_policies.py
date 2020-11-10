import kerasncp as ncp
import tensorflow as tf


class NeuralCircuitPolicies(tf.keras.Model):
    def __init__(self, output_length, inter_neurons, command_neurons, motor_neurons, sensory_fanout, inter_fanout, recurrent_command_synapses, motor_fanin):
        super(NeuralCircuitPolicies, self).__init__()
        # parameters
        self.output_length = output_length
        self.inter_neurons = inter_neurons
        self.command_neurons = command_neurons
        self.motor_neurons = motor_neurons
        self.sensory_fanout = sensory_fanout
        self.inter_fanout = inter_fanout
        self.recurrent_command_synapses = recurrent_command_synapses
        self.motor_fanin = motor_fanin
        # used layers
        self.rnn = tf.keras.layers.RNN(
            ncp.LTCCell(
                ncp.wirings.NCP(self.inter_neurons, self.command_neurons, self.motor_neurons, self.sensory_fanout, self.inter_fanout, self.recurrent_command_synapses, self.motor_fanin)),
            return_sequences=True)
        self.dense_layer = tf.keras.layers.Dense(self.output_length)

    def call(self, inputs, training=None, mask=None):
        signals, times = inputs
        rnn_output = self.rnn((signals, times))
        return self.dense_layer(rnn_output)

    def get_config(self):
        pass
