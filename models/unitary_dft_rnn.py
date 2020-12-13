class MemoryCell(tf.keras.layers.Layer):

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