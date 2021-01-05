import tensorflow as tf
import tensorflow_probability as tfp

import experiments.models.model_factory as model_factory


def get_unitary_matrix(vector):
    triangular_matrix = tfp.math.fill_triangular(vector)
    skew_hermitian_matrix = triangular_matrix - tf.linalg.adjoint(triangular_matrix)
    unitary_matrix = tf.linalg.expm(skew_hermitian_matrix)
    return unitary_matrix


def modrelu(x, bias):
    return tf.cast(tf.keras.activations.relu(tf.math.abs(x) + bias), tf.complex64) * (x / tf.cast(tf.math.abs(x), tf.complex64))


@tf.keras.utils.register_keras_serializable()
class EnhancedUnitaryRNN(tf.keras.layers.AbstractRNNCell):
    def __init__(self, state_size, output_size, **kwargs):
        super().__init__(**kwargs)
        self.state_size_value = state_size
        self.output_size_value = output_size
        self.real_state_vector = self.add_weight('real_state_vector', (self.state_size * (self.state_size + 1) // 2,), tf.float32, tf.keras.initializers.Constant())
        self.imag_state_vector = self.add_weight('imag_state_vector', (self.state_size * (self.state_size + 1) // 2,), tf.float32, tf.keras.initializers.Constant())
        self.real_initial_state = self.add_weight('real_initial_state', (1, self.state_size), tf.float32, tf.keras.initializers.Constant(1 / self.state_size ** 0.5))
        self.imag_initial_state = self.add_weight('imag_initial_state', (1, self.state_size), tf.float32, tf.keras.initializers.Constant())
        self.bias = self.add_weight('bias', (self.state_size, 1), tf.float32, tf.keras.initializers.Constant())
        self.output_layer = tf.keras.layers.Dense(self.output_size)
        self.real_input_matrix = None
        self.imag_input_matrix = None

    @property
    def state_size(self):
        return self.state_size_value

    @property
    def output_size(self):
        return self.output_size_value

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return tf.repeat(tf.complex(self.real_initial_state, self.imag_initial_state), batch_size, 0)

    def build(self, input_shape):
        inputs_size = model_factory.get_concat_input_shape(input_shape)
        self.real_input_matrix = self.add_weight('real_input_matrix', (self.state_size, 2 * inputs_size), tf.float32, tf.keras.initializers.GlorotUniform())
        self.imag_input_matrix = self.add_weight('imag_input_matrix', (self.state_size, 2 * inputs_size), tf.float32, tf.keras.initializers.Constant())

    def call(self, inputs, states):
        inputs = model_factory.get_concat_inputs(inputs)
        state_matrix = get_unitary_matrix(tf.complex(self.real_state_vector, self.imag_state_vector))
        input_matrix = tf.complex(self.real_input_matrix, self.imag_input_matrix)
        time_domain_inputs = tf.cast(inputs, tf.complex64)
        frequency_domain_inputs = tf.signal.fft(time_domain_inputs)
        input_parts = tf.matmul(input_matrix, tf.concat((time_domain_inputs, frequency_domain_inputs), -1)[..., tf.newaxis])
        state_parts = tf.matmul(state_matrix, states[0][..., tf.newaxis])
        next_states = tf.squeeze(modrelu(state_parts + input_parts, self.bias), -1)
        outputs = self.output_layer(tf.concat((tf.math.real(next_states), tf.math.imag(next_states)), -1))
        return outputs, (next_states,)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'state_size': self.state_size,
            'output_size': self.output_size,
        })
        return config
