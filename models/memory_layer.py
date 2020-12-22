import tensorflow as tf

from models.transformer import positional_encoding


class MemoryLayerCell(tf.keras.layers.AbstractRNNCell):
    def __init__(self, memory_rows=16, memory_columns=16, memory_precision=16, output_size=None, discretization_steps=2,
                 embedding_size=64, read_heads=2, write_heads=2, trainable_memory_parameters=False):
        super().__init__()
        self.memory_rows = memory_rows
        self.memory_columns = memory_columns
        self.memory_size = self.memory_rows * self.memory_columns
        self.memory_precision = memory_precision
        self.memory_cell_amount = self.memory_size * self.memory_precision
        self.state_size_value = 2 * self.memory_cell_amount
        if output_size is None:
            self.output_size_value = self.memory_size
        else:
            self.output_size_value = output_size
        self.discretization_steps = discretization_steps
        self.embedding_size = embedding_size
        self.input_embedding = tf.keras.layers.Dense(self.embedding_size)
        self.memory_embedding = tf.keras.layers.Dense(self.embedding_size)
        self.read_heads = read_heads
        self.read_attention = tf.keras.layers.MultiHeadAttention(self.read_heads, self.embedding_size)
        self.write_heads = write_heads
        self.controller = tf.keras.layers.LSTM(self.output_size + self.write_heads * self.memory_rows + self.write_heads * self.memory_columns)
        self.trainable_memory_parameters = trainable_memory_parameters
        self.params = {
            'step_size': self.add_weight(name='step_size', shape=(self.memory_cell_amount,),
                                         initializer=tf.keras.initializers.Constant(1.5573331), trainable=self.trainable_memory_parameters),
            'capacitance': self.add_weight(name='capacitance', shape=(self.memory_cell_amount,),
                                           initializer=tf.keras.initializers.Constant(1), trainable=False),
            'leakage_conductance': self.add_weight(name='leakage_conductance', shape=(self.memory_cell_amount,),
                                                   initializer=tf.keras.initializers.Constant(0.4505964), trainable=self.trainable_memory_parameters),
            'resting_potential': self.add_weight(name='resting_potential', shape=(self.memory_cell_amount,),
                                                 initializer=tf.keras.initializers.Constant(0), trainable=False),
            'recurrent_conductance': self.add_weight(name='recurrent_conductance', shape=(self.memory_cell_amount,),
                                                     initializer=tf.keras.initializers.Constant(1.0334609), trainable=self.trainable_memory_parameters),
            'recurrent_mean_conductance_potential': self.add_weight(name='recurrent_mean_conductance_potential', shape=(self.memory_cell_amount,),
                                                                    initializer=tf.keras.initializers.Constant(0.07879465), trainable=self.trainable_memory_parameters),
            'recurrent_std_conductance_potential': self.add_weight(name='recurrent_std_conductance_potential', shape=(self.memory_cell_amount,),
                                                                   initializer=tf.keras.initializers.Constant(100), trainable=False),
            'recurrent_target_potential': self.add_weight(name='recurrent_target_potential', shape=(self.memory_cell_amount,),
                                                          initializer=tf.keras.initializers.Constant(1.4378392), trainable=self.trainable_memory_parameters),
            'inhibitory_conductance': self.add_weight(name='inhibitory_conductance', shape=(self.memory_cell_amount,),
                                                      initializer=tf.keras.initializers.Constant(1.3365093), trainable=self.trainable_memory_parameters),
            'inhibitory_mean_conductance_potential': self.add_weight(name='inhibitory_mean_conductance_potential', shape=(self.memory_cell_amount,),
                                                                     initializer=tf.keras.initializers.Constant(0.06618887), trainable=self.trainable_memory_parameters),
            'inhibitory_std_conductance_potential': self.add_weight(name='inhibitory_std_conductance_potential', shape=(self.memory_cell_amount,),
                                                                    initializer=tf.keras.initializers.Constant(100), trainable=False),
            'inhibitory_target_potential': self.add_weight(name='inhibitory_target_potential', shape=(self.memory_cell_amount,),
                                                           initializer=tf.keras.initializers.Constant(0), trainable=False),
            'input_conductance': self.add_weight(name='input_conductance', shape=(self.memory_cell_amount,),
                                                 initializer=tf.keras.initializers.Constant(0.07915332), trainable=self.trainable_memory_parameters),
            'input_mean_conductance_potential': self.add_weight(name='input_mean_conductance_potential', shape=(self.memory_cell_amount,),
                                                                initializer=tf.keras.initializers.Constant(0.5), trainable=False),
            'input_std_conductance_potential': self.add_weight(name='input_std_conductance_potential', shape=(self.memory_cell_amount,),
                                                               initializer=tf.keras.initializers.Constant(100), trainable=False),
            'input_target_potential': self.add_weight(name='input_target_potential', shape=(self.memory_cell_amount,),
                                                      initializer=tf.keras.initializers.Constant(1.5931877), trainable=self.trainable_memory_parameters),
        }
        self.capacitances = self.params['capacitance'][..., tf.newaxis, tf.newaxis]
        self.partial_step_size = self.params['step_size'][..., tf.newaxis, tf.newaxis] / self.discretization_steps
        self.conductances = tf.expand_dims(tf.stack((
            self.params['input_conductance'], self.params['recurrent_conductance'],
            self.params['inhibitory_conductance'], self.params['leakage_conductance']), -1), -2)
        self.target_potentials = tf.expand_dims(tf.stack((
            self.params['input_target_potential'], self.params['recurrent_target_potential'],
            self.params['inhibitory_target_potential'], self.params['resting_potential']), -1), -2)
        self.mean_conductance_potentials = tf.expand_dims(tf.stack((
            self.params['input_mean_conductance_potential'], self.params['recurrent_mean_conductance_potential'],
            self.params['inhibitory_mean_conductance_potential'], tf.zeros((self.memory_cell_amount,))), -1), -2)
        self.std_conductance_potentials = tf.expand_dims(tf.stack((
            self.params['input_std_conductance_potential'], self.params['recurrent_std_conductance_potential'],
            self.params['inhibitory_std_conductance_potential'], tf.ones((self.memory_cell_amount,))), -1), -2)
        self.powers_of_two = [2 ** i for i in range(self.memory_precision)]
        self.positional_encoding = positional_encoding(tf.range(self.memory_rows + 1)[tf.newaxis, ..., tf.newaxis], self.embedding_size)

    @property
    def state_size(self):
        return self.state_size_value

    @property
    def output_size(self):
        return self.output_size_value

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return tf.reshape(tf.concat((tf.zeros((batch_size, self.memory_cell_amount, 1)), tf.ones((batch_size, self.memory_cell_amount, 1))), -1), (-1, self.state_size))

    def state_change(self, neuron_inputs, neuron_potentials):
        reshaped_neuron_inputs = tf.reshape(neuron_inputs, (-1, self.memory_cell_amount, 2))
        reshaped_neuron_potentials = tf.reshape(neuron_potentials, (-1, self.memory_cell_amount, 2))
        complementary_neuron_potentials = tf.roll(reshaped_neuron_potentials, 1, -1)
        presynaptic_potentials = tf.stack((
            reshaped_neuron_inputs, reshaped_neuron_potentials,
            complementary_neuron_potentials, tf.ones_like(reshaped_neuron_inputs) * float('inf')), -1)
        synaptic_currents = self.conductances * tf.sigmoid(self.std_conductance_potentials * (presynaptic_potentials - self.mean_conductance_potentials)) * (
                self.target_potentials - reshaped_neuron_potentials[..., tf.newaxis])
        state_change = tf.reshape(tf.reduce_sum(synaptic_currents / self.capacitances * self.partial_step_size, -1), (-1, self.state_size))
        return state_change

    def call(self, inputs, states):
        neuron_potentials = states[0]
        memory_row_contents = tf.reshape(neuron_potentials, (-1, self.memory_rows, 2 * self.memory_columns * self.memory_precision))[..., 0::2]
        embedded_memory_row_contents = self.memory_embedding(memory_row_contents)
        embedded_inputs = self.input_embedding(tf.expand_dims(inputs, -2))
        augmented_inputs = tf.concat((embedded_inputs, embedded_memory_row_contents), -2) + self.positional_encoding
        read_attention_output = self.read_attention(augmented_inputs, augmented_inputs)[:, :1, :]
        controller_output = self.controller(read_attention_output)
        memory_layer_outputs = controller_output[:, :self.output_size]
        neuron_inputs = tf.zeros((128, self.state_size))
        for _ in range(self.discretization_steps):
            neuron_potentials += self.state_change(neuron_inputs, neuron_potentials)
        return memory_layer_outputs, (neuron_potentials,)


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
