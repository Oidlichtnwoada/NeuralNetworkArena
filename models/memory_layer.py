import tensorflow as tf


class MemoryLayerCell(tf.keras.layers.Layer):
    def __init__(self, state_size, output_size):
        super().__init__()
        # save the state size (number of neurons) - only an even amount is allowed as neuron pairs are used for memory cells
        assert state_size % 2 == 0
        self.state_size = state_size
        # save the output_size (vector size of the output control output)
        self.output_size = output_size
        # input control - provides one input for each memory cell
        self.input_control = tf.keras.layers.Dense(self.state_size // 2)
        # output control - creates the final output of the memory layer
        self.output_control = tf.keras.layers.Dense(self.output_size)
        # create a dictionary with all trainable parameters in this layer
        self.params = {'input_scaling': self.add_weight(shape=(self.state_size,), initializer=tf.keras.initializers.Constant(1)),
                       'input_bias': self.add_weight(shape=(self.state_size,), initializer=tf.keras.initializers.Constant(0)),
                       'output_scaling': self.add_weight(shape=(self.state_size // 2,), initializer=tf.keras.initializers.Constant(1)),
                       'output_bias': self.add_weight(shape=(self.state_size // 2,), initializer=tf.keras.initializers.Constant(0)),
                       'max_conductance': self.add_weight(shape=(self.state_size, 2), initializer=tf.keras.initializers.Constant(1), constraint=tf.keras.constraints.NonNeg()),
                       'med_conductance_potential': self.add_weight(shape=(self.state_size, 2), initializer=tf.keras.initializers.Constant(0)),
                       'std_conductance': self.add_weight(shape=(self.state_size, 2), initializer=tf.keras.initializers.Constant(1), constraint=tf.keras.constraints.NonNeg())}

    def build(self, input_shape):
        pass

    def call(self, inputs, states):
        # states are passed as tuple
        states = states[0]
        # build the preliminary memory cell inputs using all available information
        preliminary_memory_cell_inputs = self.input_control(tf.concat([inputs, states], -1))
        # duplicate every entry to get an input for each neuron
        memory_cell_inputs = tf.reshape(tf.repeat(preliminary_memory_cell_inputs[..., tf.newaxis], 2, -1), (-1, self.state_size))
        # pass the memory cell inputs through an affine transformation
        affine_memory_cell_inputs = memory_cell_inputs * self.params['input_scaling'] + self.params['input_bias']
        # build a tensor representing both potentials of two neuron pairs
        neuron_pair_potentials = tf.reshape(states, (-1, self.state_size // 2, 2))
        # build the presynaptic potentials for the recurrent excitatory and the reciprocal inhibitory connection
        presynaptic_potentials = tf.reshape(tf.concat([neuron_pair_potentials, tf.roll(neuron_pair_potentials, 1, -1)], -1), (-1, self.state_size, 2))
        # compute the synaptic conductance
        synaptic_conductance = self.params['max_conductance'] / (1 + tf.math.exp(-self.params['std_conductance'] * (presynaptic_potentials - self.params['med_conductance_potential'])))
        # compute the voltage difference between resting potential (these are different for excitatory and inhibitory synapses) and the neuron's current potential
        synaptic_potential_difference = tf.concat([tf.ones_like(states[..., tf.newaxis]), -tf.ones_like(states[..., tf.newaxis])], -1) - states[..., tf.newaxis]
        # the synaptic current is the conductance multiplied with the voltage
        synaptic_memory_cell_inputs = synaptic_conductance * synaptic_potential_difference
        # compute the change in potentials of the neurons and for this sum up the synaptic currents for each neuron
        states_change = affine_memory_cell_inputs + tf.reduce_sum(synaptic_memory_cell_inputs, -1)
        # update the current memory state by using a tanh activation function after doing the state update
        next_states = tf.keras.activations.tanh(states + states_change)
        # only the output of the first memory cell is taken
        memory_cell_outputs = next_states[:, 0::2]
        # pass the memory cell outputs through an affine transformation
        affine_memory_cell_outputs = memory_cell_outputs * self.params['output_scaling'] + self.params['output_bias']
        # build the memory layer output using all available information
        memory_layer_outputs = self.output_control(tf.concat([inputs, affine_memory_cell_outputs, next_states[:, 1::2]], -1))
        # return the memory layer outputs and the next states as tuple
        return memory_layer_outputs, (next_states,)


class MemoryLayerAttention(tf.keras.layers.Layer):
    def __init__(self, dim, heads):
        super().__init__()
        # create a memory layer out of heads memory cells with size twice dim because queries and keys are concatenated together
        self.memory_layers = [tf.keras.layers.RNN(MemoryLayerCell(2 * dim, dim)) for _ in range(heads)]
        # create a dense layer to merge all heads and the concatenated representation to size dim
        self.dense_layer = tf.keras.layers.Dense(dim)

    def compute_accumulated_representation(self, query, values):
        # concatenate query at query_index to each value to create the input for memory layers
        duplicated_query = tf.repeat(query, values.shape[1], axis=1)
        memory_layer_input = tf.concat([duplicated_query, values], axis=-1)
        # accumulate information with memory layers for each head and concatenate the outputs together
        accumulated_inputs = tf.concat([memory_layer(memory_layer_input) for memory_layer in self.memory_layers], axis=-1)
        # merge outputs from all heads to size dim via a dense layer and add a dimension for later concatenation
        return tf.expand_dims(self.dense_layer(accumulated_inputs), axis=1)

    def call(self, inputs, **kwargs):
        # split inputs tuple to the arguments
        queries, _, values, _ = inputs
        # compute an accumulated representation for all queries
        return tf.concat([self.compute_accumulated_representation(queries[:, query_index:query_index + 1, :], values) for query_index in range(queries.shape[1])], axis=1), None
