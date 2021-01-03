import tensorflow as tf

import experiments.models.ct_rnn as ct_rnn


@tf.keras.utils.register_keras_serializable()
class ODELSTM(tf.keras.layers.AbstractRNNCell):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.state_size_value = (units, units)
        self.initializer = "glorot_uniform"
        self.recurrent_initializer = "orthogonal"
        self.ctrnn = ct_rnn.CTRNNCell(self.units, num_unfolds=4, method="euler")
        self.input_kernel, self.recurrent_kernel, self.bias = (None,) * 3

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return (
            tf.zeros([batch_size, self.units], dtype=tf.float32),
            tf.zeros([batch_size, self.units], dtype=tf.float32),
        )

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if isinstance(input_shape[0], tuple):
            input_dim = input_shape[0][-1]
        self.ctrnn.build([self.units])
        self.input_kernel = self.add_weight(
            shape=(input_dim, 4 * self.units),
            initializer=self.initializer,
            name="input_kernel",
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, 4 * self.units),
            initializer=self.recurrent_initializer,
            name="recurrent_kernel",
        )
        self.bias = self.add_weight(
            shape=(4 * self.units),
            initializer=tf.keras.initializers.Zeros(),
            name="bias",
        )
        self.built = True

    def call(self, inputs, states):
        cell_state, ode_state = states
        elapsed = 1.0
        if (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) > 1:
            elapsed = inputs[1]
            inputs = inputs[0]
        z = (
                tf.matmul(inputs, self.input_kernel)
                + tf.matmul(ode_state, self.recurrent_kernel)
                + self.bias
        )
        i, ig, fg, og = tf.split(z, 4, axis=-1)
        input_activation = tf.nn.tanh(i)
        input_gate = tf.nn.sigmoid(ig)
        forget_gate = tf.nn.sigmoid(fg + 3.0)
        output_gate = tf.nn.sigmoid(og)
        new_cell = cell_state * forget_gate + input_activation * input_gate
        ode_input = tf.nn.tanh(new_cell) * output_gate
        ode_output, new_ode_state = self.ctrnn.call([ode_input, elapsed], [ode_state])
        return ode_output, [new_cell, new_ode_state[0]]

    @property
    def state_size(self):
        return self.state_size_value

    @property
    def output_size(self):
        return self.state_size_value[0]

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units': self.units
        })
        return config
