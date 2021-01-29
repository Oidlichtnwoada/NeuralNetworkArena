import tensorflow as tf

import experiments.models.neural_circuit_policies as ncp
import experiments.models.unitary_rnn as urnn


@tf.keras.utils.register_keras_serializable()
class UnitaryNCP(tf.keras.layers.Layer):
    def __init__(self, units_urnn, units_ncp, output_size, return_sequences=False, **kwargs):
        super().__init__(**kwargs)
        self.units_urnn = units_urnn
        self.units_ncp = units_ncp
        self.output_size = output_size
        self.return_sequences = return_sequences
        self.urnn = tf.keras.layers.RNN(urnn.EUNNCell(units_urnn), return_sequences=True)
        self.ncp = ncp.NeuralCircuitPolicies(units_ncp, self.output_size, return_sequences=self.return_sequences)

    def call(self, inputs, **kwargs):
        urnn_output = self.urnn(inputs)
        urnn_output_real = tf.concat((tf.math.real(urnn_output), tf.math.imag(urnn_output)), -1)
        ncp_output = self.ncp(urnn_output_real)
        return ncp_output

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units_urnn': self.units_urnn,
            'units_ncp': self.units_ncp,
            'output_size': self.output_size,
            'return_sequences': self.return_sequences,
        })
        return config
