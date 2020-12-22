import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


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


batch_size_value = 32
epochs = 16
memory_high_symbol = 1
memory_low_symbol = 0
memory_length = 128
cell_switches = 2
sample_batches = 32
sample_size = batch_size_value * sample_batches
learning_rate = 1E-3
weights_directory = '../weights/memory_cell/checkpoint'
use_saved_weights = False
run_eagerly = False

model_input = np.zeros((sample_size, (cell_switches + 1) * memory_length, 2))
model_output = np.zeros((sample_size, (cell_switches + 1) * memory_length, 2))
for i in range(cell_switches + 1):
    even = int(i % 2 == 0)
    odd = int(i % 2 == 1)
    model_input[0::2, i * memory_length, odd] = memory_high_symbol
    model_input[0::2, i * memory_length, even] = memory_low_symbol
    model_output[0::2, i * memory_length:(i + 1) * memory_length, 0] = even * memory_high_symbol
    model_output[0::2, i * memory_length:(i + 1) * memory_length, 1] = odd * memory_high_symbol
    model_input[1::2, i * memory_length, even] = memory_high_symbol
    model_input[1::2, i * memory_length, odd] = memory_low_symbol
    model_output[1::2, i * memory_length:(i + 1) * memory_length, 0] = odd * memory_high_symbol
    model_output[1::2, i * memory_length:(i + 1) * memory_length, 1] = even * memory_high_symbol

input_tensor = tf.keras.Input(shape=((cell_switches + 1) * memory_length, 2), batch_size=batch_size_value)
output_tensor = tf.keras.layers.RNN(MemoryCell(), return_sequences=True)(input_tensor)
model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=tf.keras.losses.MeanSquaredError(), run_eagerly=run_eagerly)
if use_saved_weights:
    model.load_weights(weights_directory).expect_partial()
model.fit(x=model_input, y=model_output, batch_size=batch_size_value, epochs=epochs, validation_data=(model_input[:batch_size_value], model_output[:batch_size_value]),
          callbacks=[tf.keras.callbacks.ModelCheckpoint(weights_directory, save_best_only=True, save_weights_only=True)])

sample_outputs = model(model_input[:32])
plt.plot(sample_outputs[0, :, 0], label='cell state')
plt.plot(model_output[0, :, 0], label='expected cell state')
plt.legend()
plt.title('starting with saving 1')
plt.show()
plt.plot(sample_outputs[0, :, 1], label='second neuron state')
plt.plot(model_output[0, :, 1], label='expected second neuron state')
plt.legend()
plt.title('starting with saving 1')
plt.show()
plt.plot(sample_outputs[1, :, 0], label='cell state')
plt.plot(model_output[1, :, 0], label='expected cell state')
plt.legend()
plt.title('starting with saving 0')
plt.show()
plt.plot(sample_outputs[1, :, 1], label='second neuron state')
plt.plot(model_output[1, :, 1], label='expected second neuron state')
plt.legend()
plt.title('starting with saving 0')
plt.show()
