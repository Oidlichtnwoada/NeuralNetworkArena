import tensorflow as tf


class RecurrentMemoryCell(tf.keras.layers.Layer):
    def __init__(self, state_size, output_size):
        super().__init__()
        # save the state size
        self.state_size = state_size
        # save the output_size (vector size of the output control output)
        self.output_size = output_size
        # input control - provides one input for each memory cell
        self.input_control = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self.state_size * 4),
            tf.keras.layers.Dense(self.state_size * 2)])
        # output control - creates the final output of the memory layer
        self.output_control = tf.keras.models.Sequential([
            tf.keras.layers.Dense(2 * self.output_size),
            tf.keras.layers.Dense(self.output_size)])

    def call(self, inputs, states):
        # states are passed as a tuple
        state = states[0]
        # build the preliminary memory cell signals using all available information
        preliminary_memory_cell_signals = self.input_control(tf.concat([inputs, state], -1))
        # split up the preliminary memory cell signals into control and data
        reshaped_memory_cell_signals = tf.reshape(preliminary_memory_cell_signals, (-1, self.state_size, 2))
        # create data and control signals
        memory_cell_data_signals = reshaped_memory_cell_signals[:, :, 0]
        memory_cell_control_signals = reshaped_memory_cell_signals[:, :, 1]
        # if some control signal is greater than the threshold assign the data signal to the state
        next_state = tf.keras.activations.sigmoid(memory_cell_control_signals) * memory_cell_data_signals + tf.keras.activations.sigmoid(-memory_cell_control_signals) * state
        # build the memory layer output using only the outputs of the memory cells
        memory_layer_outputs = self.output_control(next_state)
        # return the memory layer outputs and the next states as tuple
        return memory_layer_outputs, (next_state,)
