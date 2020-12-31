import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class MemoryCell(tf.keras.layers.AbstractRNNCell):
    def __init__(self, discretization_steps=2):
        super().__init__()
        self.discretization_steps = discretization_steps
        self.params = {
            'step_size': self.add_weight(name='step_size', shape=(1,), initializer=tf.keras.initializers.Constant(1.5573331)),
            'capacitance': self.add_weight(name='capacitance', shape=(1,), initializer=tf.keras.initializers.Constant(1), trainable=False),
            'leakage_conductance': self.add_weight(name='leakage_conductance', shape=(1,), initializer=tf.keras.initializers.Constant(0.4505964)),
            'resting_potential': self.add_weight(name='resting_potential', shape=(1,), initializer=tf.keras.initializers.Constant(0), trainable=False),
            'recurrent_conductance': self.add_weight(name='recurrent_conductance', shape=(1,), initializer=tf.keras.initializers.Constant(1.0334609)),
            'recurrent_mean_conductance_potential': self.add_weight(name='recurrent_mean_conductance_potential', shape=(1,), initializer=tf.keras.initializers.Constant(0.07879465)),
            'recurrent_std_conductance_potential': self.add_weight(name='recurrent_std_conductance_potential', shape=(1,), initializer=tf.keras.initializers.Constant(100), trainable=False),
            'recurrent_target_potential': self.add_weight(name='recurrent_target_potential', shape=(1,), initializer=tf.keras.initializers.Constant(1.4378392)),
            'inhibitory_conductance': self.add_weight(name='inhibitory_conductance', shape=(1,), initializer=tf.keras.initializers.Constant(1.3365093)),
            'inhibitory_mean_conductance_potential': self.add_weight(name='inhibitory_mean_conductance_potential', shape=(1,), initializer=tf.keras.initializers.Constant(0.06618887)),
            'inhibitory_std_conductance_potential': self.add_weight(name='inhibitory_std_conductance_potential', shape=(1,), initializer=tf.keras.initializers.Constant(100), trainable=False),
            'inhibitory_target_potential': self.add_weight(name='inhibitory_target_potential', shape=(1,), initializer=tf.keras.initializers.Constant(0), trainable=False),
            'input_conductance': self.add_weight(name='input_conductance', shape=(1,), initializer=tf.keras.initializers.Constant(0.07915332)),
            'input_mean_conductance_potential': self.add_weight(name='input_mean_conductance_potential', shape=(1,), initializer=tf.keras.initializers.Constant(0.5), trainable=False),
            'input_std_conductance_potential': self.add_weight(name='input_std_conductance_potential', shape=(1,), initializer=tf.keras.initializers.Constant(100), trainable=False),
            'input_target_potential': self.add_weight(name='input_target_potential', shape=(1,), initializer=tf.keras.initializers.Constant(1.5931877)),
        }

    @property
    def state_size(self):
        return 2

    @property
    def output_size(self):
        return 2

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return tf.concat((tf.zeros((batch_size, 1)), tf.ones((batch_size, 1))), -1)

    def synaptic_current(self, synapse_type, presynaptic, postsynaptic):
        conductance = self.params[f'{synapse_type}_conductance'] * \
                      tf.math.sigmoid(self.params[f'{synapse_type}_std_conductance_potential'] * (presynaptic - self.params[f'{synapse_type}_mean_conductance_potential']))
        potential_difference = self.params[f'{synapse_type}_target_potential'] - postsynaptic
        return conductance * potential_difference

    def leakage_current(self, potential):
        return self.params['leakage_conductance'] * (self.params['resting_potential'] - potential)

    def state_derivative(self, neuron_x_inputs, neuron_x_potentials, neuron_y_inputs, neuron_y_potentials):
        current_x = self.synaptic_current('input', neuron_x_inputs, neuron_x_potentials)
        current_x += self.synaptic_current('inhibitory', neuron_y_potentials, neuron_x_potentials)
        current_x += self.synaptic_current('recurrent', neuron_x_potentials, neuron_x_potentials)
        current_x += self.leakage_current(neuron_x_potentials)
        current_y = self.synaptic_current('input', neuron_y_inputs, neuron_y_potentials)
        current_y += self.synaptic_current('inhibitory', neuron_x_potentials, neuron_y_potentials)
        current_y += self.synaptic_current('recurrent', neuron_y_potentials, neuron_y_potentials)
        current_y += self.leakage_current(neuron_y_potentials)
        return current_x / self.params['capacitance'], current_y / self.params['capacitance']

    def call(self, inputs, states):
        neuron_x_inputs = inputs[:, :1]
        neuron_y_inputs = inputs[:, 1:]
        neuron_x_potentials = states[0][:, :1]
        neuron_y_potentials = states[0][:, 1:]
        partial_step_size = self.params['step_size'] / self.discretization_steps
        for _ in range(self.discretization_steps):
            neuron_x_potentials_derivative, neuron_y_potentials_derivative = self.state_derivative(neuron_x_inputs, neuron_x_potentials, neuron_y_inputs, neuron_y_potentials)
            neuron_x_potentials += neuron_x_potentials_derivative * partial_step_size
            neuron_y_potentials += neuron_y_potentials_derivative * partial_step_size
        states = tf.concat((neuron_x_potentials, neuron_y_potentials), axis=-1)
        return states, (states,)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'discretization_steps': self.discretization_steps
        })
        return config
