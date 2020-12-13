import tensorflow as tf


def get_unitary_matrix(matrix):
    return tf.linalg.expm(1j * (matrix + tf.linalg.adjoint(matrix)))


def modrelu(x, bias):
    return tf.keras.activations.relu(tf.math.abs(x) + bias) * (x / tf.math.abs(x))


class EnhancedUnitaryRNN(tf.keras.layers.Layer):
    def __init__(self, state_size, output_size):
        super().__init__(dtype=tf.complex64)
        self.state_size = state_size
        self.output_size = output_size
        self.input_matrix = None
        self.state_matrix = self.add_weight('state_matrix', (self.state_size, self.state_size), tf.complex64, tf.keras.initializers.Identity)
        self.step_matrix = self.add_weight('step_matrix', (self.state_size, self.state_size), tf.complex64, tf.keras.initializers.Identity)
        self.first_bias = self.add_weight('first_bias', (1,), tf.float32, tf.keras.initializers.constant())
        self.second_bias = self.add_weight('second_bias', (1,), tf.float32, tf.keras.initializers.constant())
        self.output_layer = tf.keras.layers.Dense(self.output_size)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return tf.ones((batch_size, self.state_size), dtype)

    def build(self, input_shape):
        self.input_matrix = self.add_weight('input_matrix', (self.state_size, 2 * input_shape[-1]), tf.complex64, tf.keras.initializers.Identity)

    def call(self, inputs, states):
        states = states[0]
        time_domain_inputs = tf.cast(inputs, tf.complex64)
        frequency_domain_inputs = tf.signal.fft(time_domain_inputs)
        input_parts = tf.matmul(self.input_matrix, tf.concat((time_domain_inputs, frequency_domain_inputs), -1)[..., tf.newaxis])
        state_parts = tf.matmul(get_unitary_matrix(self.state_matrix), states[..., tf.newaxis])
        preliminary_next_states = modrelu(state_parts + input_parts, self.first_bias)
        next_states = modrelu(tf.matmul(get_unitary_matrix(self.step_matrix), preliminary_next_states), self.second_bias)
        output = self.output_layer(tf.concat(tf.math.real(next_states), tf.math.imag(next_states), -1))
        return output, (next_states,)
