import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class MemoryCell(tf.keras.layers.Layer):
    def __init__(self, use_recurrent_connections=True):
        super().__init__()
        self.state_size = 2
        self.output_size = 2
        self.use_recurrent_connections = use_recurrent_connections
        self.params = {
            'capacitance_x': self.add_weight(name='capacitance_x', shape=(1,), initializer=tf.keras.initializers.Constant(1)),
            'capacitance_y': self.add_weight(name='capacitance_y', shape=(1,), initializer=tf.keras.initializers.Constant(1)),
            'conductance_ax': self.add_weight(name='conductance_ax', shape=(1,), initializer=tf.keras.initializers.Constant(1E-3)),
            'mean_ax': self.add_weight(name='mean_ax', shape=(1,), initializer=tf.keras.initializers.Constant(3E-1)),
            'std_ax': self.add_weight(name='std_ax', shape=(1,), initializer=tf.keras.initializers.Constant(3)),
            'pot_ax': self.add_weight(name='pot_ax', shape=(1,), initializer=tf.keras.initializers.Constant(1)),
            'conductance_by': self.add_weight(name='conductance_by', shape=(1,), initializer=tf.keras.initializers.Constant(1E-3)),
            'mean_by': self.add_weight(name='mean_by', shape=(1,), initializer=tf.keras.initializers.Constant(3E-1)),
            'std_by': self.add_weight(name='std_by', shape=(1,), initializer=tf.keras.initializers.Constant(3)),
            'pot_by': self.add_weight(name='pot_by', shape=(1,), initializer=tf.keras.initializers.Constant(1)),
            'conductance_xy': self.add_weight(name='conductance_xy', shape=(1,), initializer=tf.keras.initializers.Constant(1E-3)),
            'mean_xy': self.add_weight(name='mean_xy', shape=(1,), initializer=tf.keras.initializers.Constant(3E-1)),
            'std_xy': self.add_weight(name='std_xy', shape=(1,), initializer=tf.keras.initializers.Constant(3)),
            'pot_xy': self.add_weight(name='pot_xy', shape=(1,), initializer=tf.keras.initializers.Constant(1)),
            'conductance_yx': self.add_weight(name='conductance_yx', shape=(1,), initializer=tf.keras.initializers.Constant(1E-3)),
            'mean_yx': self.add_weight(name='mean_yx', shape=(1,), initializer=tf.keras.initializers.Constant(3E-1)),
            'std_yx': self.add_weight(name='std_yx', shape=(1,), initializer=tf.keras.initializers.Constant(3)),
            'pot_yx': self.add_weight(name='pot_yx', shape=(1,), initializer=tf.keras.initializers.Constant(1))}
        if self.use_recurrent_connections:
            additional_parameters = {
                'conductance_xx': self.add_weight(name='conductance_xx', shape=(1,), initializer=tf.keras.initializers.Constant(1E-3)),
                'mean_xx': self.add_weight(name='mean_xx', shape=(1,), initializer=tf.keras.initializers.Constant(3E-1)),
                'std_xx': self.add_weight(name='std_xx', shape=(1,), initializer=tf.keras.initializers.Constant(3)),
                'pot_xx': self.add_weight(name='pot_xx', shape=(1,), initializer=tf.keras.initializers.Constant(1)),
                'conductance_yy': self.add_weight(name='conductance_yy', shape=(1,), initializer=tf.keras.initializers.Constant(1E-3)),
                'mean_yy': self.add_weight(name='mean_yy', shape=(1,), initializer=tf.keras.initializers.Constant(3E-1)),
                'std_yy': self.add_weight(name='std_yy', shape=(1,), initializer=tf.keras.initializers.Constant(3)),
                'pot_yy': self.add_weight(name='pot_yy', shape=(1,), initializer=tf.keras.initializers.Constant(1))}
            self.params = {**self.params, **additional_parameters}

    @staticmethod
    def get_initial_state(**kwargs):
        return tf.concat((tf.zeros((batch_size, 1)), tf.ones((batch_size, 1))), -1)

    def synaptic_current(self, direction, presynaptic, postsynaptic):
        conductance = self.params[f'conductance_{direction}'] * tf.math.sigmoid(self.params[f'std_{direction}'] * (presynaptic - self.params[f'mean_{direction}']))
        potential_difference = self.params[f'pot_{direction}'] - postsynaptic
        return conductance * potential_difference

    def call(self, inputs, states):
        neuron_x_potentials = states[0][:, :1]
        neuron_y_potentials = states[0][:, 1:]
        neuron_x_inputs_a = inputs[:, :1]
        neuron_y_inputs_b = inputs[:, 1:]
        synaptic_current_x = self.synaptic_current('ax', neuron_x_inputs_a, neuron_x_potentials)
        synaptic_current_x += self.synaptic_current('yx', neuron_y_potentials, neuron_x_potentials)
        synaptic_current_y = self.synaptic_current('by', neuron_y_inputs_b, neuron_y_potentials)
        synaptic_current_y += self.synaptic_current('xy', neuron_x_potentials, neuron_y_potentials)
        if self.use_recurrent_connections:
            synaptic_current_x += self.synaptic_current('xx', neuron_x_potentials, neuron_x_potentials)
            synaptic_current_y += self.synaptic_current('yy', neuron_y_potentials, neuron_y_potentials)
        neuron_x_potentials += synaptic_current_x / self.params['capacitance_x']
        neuron_y_potentials += synaptic_current_y / self.params['capacitance_y']
        states = tf.concat((neuron_x_potentials, neuron_y_potentials), axis=-1)
        return states, (states,)


batch_size = 32
epochs = 8
memory_high_symbol = 1
memory_low_symbol = 0
memory_length = 128
cell_switches = 2
sample_batches = 32
sample_size = batch_size * sample_batches
learning_rate = 1E-3
weights_directory = '../weights/memory_cell/checkpoint'
use_saved_weights = False
run_eagerly = False
ignore_second_neuron_potential = False

loss_object = tf.keras.losses.MeanSquaredError()
model_input = np.ones((sample_size, (cell_switches + 1) * memory_length, 2)) * memory_low_symbol
model_output = np.ones((sample_size, (cell_switches + 1) * memory_length, 2)) * memory_low_symbol
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


def custom_loss(y_true, y_pred):
    if ignore_second_neuron_potential:
        y_true = y_true[..., :1]
        y_pred = y_pred[..., :1]
    return loss_object(y_true, y_pred)


input_tensor = tf.keras.Input(shape=((cell_switches + 1) * memory_length, 2), batch_size=batch_size)
output_tensor = tf.keras.layers.RNN(MemoryCell(use_recurrent_connections=False), return_sequences=True)(input_tensor)
model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=custom_loss, run_eagerly=run_eagerly)
if use_saved_weights:
    model.load_weights(weights_directory).expect_partial()
model.fit(x=model_input, y=model_output, batch_size=batch_size, epochs=epochs, validation_data=(model_input[:batch_size], model_output[:batch_size]),
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
plt.title('starting with saving 0')
plt.show()
