import tensorflow as tf


class MemoryLayerCell(tf.keras.layers.Layer):
    def __init__(self, state_size, output_size):
        super().__init__()
        # save the state size (number of neurons) - only an even amount is allowed as neuron pairs are used for memory cells
        assert state_size % 2 == 0
        self.state_size = state_size
        # save the output_size (vector size of the output control output)
        self.output_size = output_size
        # add a normalization layer for the neuron potentials
        self.normalization = tf.keras.layers.LayerNormalization(epsilon=1E-6)
        # input control - provides one input for each memory cell
        self.input_control = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self.state_size),
            tf.keras.layers.Dense(self.state_size // 2)])
        # output control - creates the final output of the memory layer
        self.output_control = tf.keras.models.Sequential([
            tf.keras.layers.Dense(2 * self.output_size),
            tf.keras.layers.Dense(self.output_size)])
        # create a dictionary with all trainable parameters in this layer
        self.params = {'excitatory_potential': self.add_weight(name='capacitance', shape=(self.state_size,),
                                                               initializer=tf.keras.initializers.Constant(1), constraint=tf.keras.constraints.MinMaxNorm(1, float('inf'))),
                       'inhibitory_potential': self.add_weight(name='capacitance', shape=(self.state_size,),
                                                               initializer=tf.keras.initializers.Constant(-1), constraint=tf.keras.constraints.MinMaxNorm(-float('inf'), -1)),
                       'capacitance': self.add_weight(name='capacitance', shape=(self.state_size,), initializer=tf.keras.initializers.Constant(1), constraint=tf.keras.constraints.NonNeg()),
                       'max_conductance': self.add_weight(name='max_conductance', shape=(self.state_size, 2), initializer=tf.keras.initializers.Constant(1), constraint=tf.keras.constraints.NonNeg()),
                       'mean_conductance_potential': self.add_weight(name='mean_conductance_potential', shape=(self.state_size, 2), initializer=tf.keras.initializers.Constant(0)),
                       'std_conductance': self.add_weight(name='std_conductance', shape=(self.state_size, 2), initializer=tf.keras.initializers.Constant(1), constraint=tf.keras.constraints.NonNeg())}

    def call(self, inputs, states):
        # states are passed as tuple
        states = states[0]
        if isinstance(inputs, tuple):
            # save time intervals if provided for irregularly sampled time series
            inputs, intervals = inputs
        else:
            # generate intervals of only ones for regularly sampled time series if no intervals are provided
            intervals = tf.ones_like(inputs)[:, :1]
        # build the preliminary memory cell inputs using all available information
        preliminary_memory_cell_inputs = self.input_control(tf.concat([inputs, states], -1))
        # duplicate every entry and negate it to get the positive and negative input for each neuron pair
        memory_cell_inputs = tf.reshape(tf.concat([preliminary_memory_cell_inputs[..., tf.newaxis], -preliminary_memory_cell_inputs[..., tf.newaxis]], -1), (-1, self.state_size))
        # build a tensor representing both potentials of a neuron pair in the last dimension
        neuron_pair_potentials = tf.reshape(states, (-1, self.state_size // 2, 2))
        # build the presynaptic potentials for the recurrent excitatory and the reciprocal inhibitory connection
        presynaptic_potentials = tf.reshape(tf.concat([neuron_pair_potentials, tf.roll(neuron_pair_potentials, 1, -1)], -1), (-1, self.state_size, 2))
        # compute the synaptic conductance
        synaptic_conductance = self.params['max_conductance'] / (1 + tf.math.exp(-self.params['std_conductance'] * (presynaptic_potentials - self.params['mean_conductance_potential'])))
        # compute the potential difference between resting potential (these are different for excitatory and inhibitory synapses) and the neuron's current potential
        synaptic_potential_difference = tf.concat([self.params['excitatory_potential'][..., tf.newaxis], self.params['inhibitory_potential'][..., tf.newaxis]], -1) - states[..., tf.newaxis]
        # the synaptic current is the conductance multiplied with the voltage
        synaptic_memory_cell_inputs = synaptic_conductance * synaptic_potential_difference
        # compute the change in potentials of the neurons by computing the gradient and multiplying it with the time difference
        states_change = (memory_cell_inputs + tf.reduce_sum(synaptic_memory_cell_inputs, -1)) / self.params['capacitance'] * intervals
        # update the current memory state by applying the changes to the current state and normalize the results
        next_states = self.normalization(states + states_change)
        # only the output of the first memory cell is taken
        memory_cell_outputs = next_states[:, 0::2]
        # build the memory layer output using only the outputs of the memory cells
        memory_layer_outputs = self.output_control(memory_cell_outputs)
        # return the memory layer outputs and the next states as tuple
        return memory_layer_outputs, (next_states,)


class MemoryLayerAttention(tf.keras.layers.Layer):
    def __init__(self, dim, heads):
        super().__init__()
        # save the dimension and the heads of the transformer
        self.dim = dim
        self.heads = heads
        # create a memory layer out of heads times dim memory cells and an output size of dim
        self.memory_layer = tf.keras.layers.RNN(MemoryLayerCell(2 * self.heads * self.dim, self.dim))

    def call(self, inputs, **kwargs):
        # split inputs tuple to the arguments
        queries, _, values, _ = inputs
        # bring queries and values to the same shape
        duplicated_queries = tf.repeat(tf.expand_dims(queries, 2), values.shape[1], 2)
        duplicated_values = tf.repeat(tf.expand_dims(values, 1), queries.shape[1], 1)
        # concatenate queries and values together and reshape it to a single batch dimension
        memory_layer_input = tf.reshape(tf.concat([duplicated_queries, duplicated_values], -1), (-1, values.shape[1], 2 * self.dim))
        # accumulate information with memory layer
        accumulated_inputs = self.memory_layer(memory_layer_input)
        # reshape the output to the right batch size and the right query dimension
        return tf.reshape(accumulated_inputs, (-1, queries.shape[1], self.dim)), None
