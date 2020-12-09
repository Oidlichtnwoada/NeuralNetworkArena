import numpy as np
import tensorflow as tf


class MemoryCell(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.state_size = 2
        self.output_size = 1
        self.params = {'capacitance_x': self.add_weight(name='capacitance_x', shape=(1,), initializer=tf.keras.initializers.Constant(4E-1)),
                       'capacitance_y': self.add_weight(name='capacitance_y', shape=(1,), initializer=tf.keras.initializers.Constant(4E-1)),
                       'conductance_ax': self.add_weight(name='conductance_ax', shape=(1,), initializer=tf.keras.initializers.Constant(1E-3)),
                       'mean_ax': self.add_weight(name='mean_ax', shape=(1,), initializer=tf.keras.initializers.Constant(3E-1)),
                       'std_ax': self.add_weight(name='std_ax', shape=(1,), initializer=tf.keras.initializers.Constant(3)),
                       'pot_ax': self.add_weight(name='pot_ax', shape=(1,), initializer=tf.keras.initializers.Constant(1)),
                       'conductance_by': self.add_weight(name='conductance_by', shape=(1,), initializer=tf.keras.initializers.Constant(1E-3)),
                       'mean_by': self.add_weight(name='mean_by', shape=(1,), initializer=tf.keras.initializers.Constant(3E-1)),
                       'std_by': self.add_weight(name='std_by', shape=(1,), initializer=tf.keras.initializers.Constant(3)),
                       'pot_by': self.add_weight(name='pot_by', shape=(1,), initializer=tf.keras.initializers.Constant(1)),
                       'conductance_xx': self.add_weight(name='conductance_xx', shape=(1,), initializer=tf.keras.initializers.Constant(1E-3)),
                       'mean_xx': self.add_weight(name='mean_xx', shape=(1,), initializer=tf.keras.initializers.Constant(3E-1)),
                       'std_xx': self.add_weight(name='std_xx', shape=(1,), initializer=tf.keras.initializers.Constant(3)),
                       'pot_xx': self.add_weight(name='pot_xx', shape=(1,), initializer=tf.keras.initializers.Constant(1)),
                       'conductance_xy': self.add_weight(name='conductance_xy', shape=(1,), initializer=tf.keras.initializers.Constant(1E-3)),
                       'mean_xy': self.add_weight(name='mean_xy', shape=(1,), initializer=tf.keras.initializers.Constant(3E-1)),
                       'std_xy': self.add_weight(name='std_xy', shape=(1,), initializer=tf.keras.initializers.Constant(3)),
                       'pot_xy': self.add_weight(name='pot_xy', shape=(1,), initializer=tf.keras.initializers.Constant(-1)),
                       'conductance_yx': self.add_weight(name='conductance_yx', shape=(1,), initializer=tf.keras.initializers.Constant(1E-3)),
                       'mean_yx': self.add_weight(name='mean_yx', shape=(1,), initializer=tf.keras.initializers.Constant(3E-1)),
                       'std_yx': self.add_weight(name='std_yx', shape=(1,), initializer=tf.keras.initializers.Constant(3)),
                       'pot_yx': self.add_weight(name='pot_yx', shape=(1,), initializer=tf.keras.initializers.Constant(-1)),
                       'conductance_yy': self.add_weight(name='conductance_yy', shape=(1,), initializer=tf.keras.initializers.Constant(1E-3)),
                       'mean_yy': self.add_weight(name='mean_yy', shape=(1,), initializer=tf.keras.initializers.Constant(3E-1)),
                       'std_yy': self.add_weight(name='std_yy', shape=(1,), initializer=tf.keras.initializers.Constant(3)),
                       'pot_yy': self.add_weight(name='pot_yy', shape=(1,), initializer=tf.keras.initializers.Constant(1))}

    def synaptic_current(self, direction, presynaptic, postsynaptic):
        conductance = self.params[f'conductance_{direction}'] * tf.math.sigmoid(self.params[f'std_{direction}'] * (presynaptic - self.params[f'mean_{direction}']))
        potential_difference = self.params[f'pot_{direction}'] - postsynaptic
        return conductance * potential_difference

    def call(self, inputs, states):
        neuron_x_potentials = states[0][:, :1]
        neuron_y_potentials = states[0][:, 1:]
        neuron_x_inputs_a = inputs[:, :1]
        neuron_y_inputs_b = inputs[:, 1:]
        synaptic_current_x = self.synaptic_current('ax', neuron_x_inputs_a, neuron_x_potentials) + \
                             self.synaptic_current('xx', neuron_x_potentials, neuron_x_potentials) + \
                             self.synaptic_current('yx', neuron_y_potentials, neuron_x_potentials)
        synaptic_current_y = self.synaptic_current('by', neuron_y_inputs_b, neuron_y_potentials) + \
                             self.synaptic_current('yy', neuron_y_potentials, neuron_y_potentials) + \
                             self.synaptic_current('xy', neuron_x_potentials, neuron_y_potentials)
        neuron_x_potentials += synaptic_current_x / self.params['capacitance_x']
        neuron_y_potentials += synaptic_current_y / self.params['capacitance_y']
        states = tf.concat((neuron_x_potentials, neuron_y_potentials), axis=-1)
        return neuron_x_potentials, (states,)


batch_size = 32
epochs = 64
memory_symbol = 1
memory_length = 100
cell_switches = 1
sample_size = batch_size * 32
learning_rate = 1E-6
weights_directory = '../weights/memory_cell'
use_saved_weights = True
run_eagerly = False
model_input = np.zeros((sample_size, (cell_switches + 1) * memory_length, 2))
model_output = np.zeros((sample_size, (cell_switches + 1) * memory_length, 1))
for i in range(cell_switches + 1):
    model_input[:, i * memory_length, int(i % 2 == 1)] = memory_symbol
    model_output[:, i * memory_length:(i + 1) * memory_length, :] = int(i % 2 == 0) * memory_symbol
input_tensor = tf.keras.Input(shape=((cell_switches + 1) * memory_length, 2), batch_size=batch_size)
output_tensor = tf.keras.layers.RNN(MemoryCell(), return_sequences=True)(input_tensor)
model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=tf.keras.losses.MeanSquaredError(), run_eagerly=run_eagerly)
if use_saved_weights:
    model.load_weights(weights_directory).expect_partial()
model.fit(x=model_input, y=model_output, batch_size=batch_size, epochs=epochs, validation_data=(model_input[:batch_size], model_output[:batch_size]),
          callbacks=[tf.keras.callbacks.ModelCheckpoint(weights_directory, save_best_only=True, save_weights_only=True)])
