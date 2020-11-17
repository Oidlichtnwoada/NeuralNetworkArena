import kerasncp as ncp
import tensorflow as tf


class MemoryCellWiring(ncp.wirings.Wiring):
    def __init__(self, dim):
        # input_dim = output_dim = dim
        self.dim = dim
        # allocate data structures for dim neuron pairs
        super().__init__(units=2 * self.dim)
        # the output dimension is half the state dimension of all neuron pairs
        self.set_output_dim(output_dim=self.dim)
        # create inhibitory recurrent connections to each neuron
        for src in range(self.units):
            self.add_synapse(src, src, polarity=-1)
        # create inhibitory connections between each neuron pair
        for src in range(self.units // 2):
            self.add_synapse(src, src + self.dim, polarity=-1)
            self.add_synapse(src + self.dim, src, polarity=-1)

    def build(self, input_shape):
        super().build(input_shape)
        # check if input dimension matches expected dimension
        assert self.input_dim == self.dim
        # connect each input with each neuron pair
        for src in range(self.input_dim):
            self.add_sensory_synapse(src, src, polarity=1)
            self.add_sensory_synapse(src, src + self.dim, polarity=1)


class MemoryCell(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        # create a memory cell (RNN) using the LTCCell class and the MemoryCellWiring
        self.memory_cell = tf.keras.layers.RNN(ncp.LTCCell(MemoryCellWiring(dim)))

    def call(self, inputs, **kwargs):
        # forward input to the memory cell
        return self.memory_cell(inputs)


class MemoryLayer(tf.keras.layers.Layer):
    def __init__(self, dim, heads=1):
        super().__init__()
        # create a memory layer out of heads memory cells with size twice dim because queries and keys are concatenated together
        self.memory_layer = [MemoryCell(2 * dim) for _ in range(heads)]
        # create a dense layer to merge all heads and the concatenated representation to size dim
        self.dense_layer = tf.keras.layers.Dense(dim)

    def compute_accumulated_representation(self, query, values):
        # concatenate query at query_index to each value to create the input for the memory cells
        duplicated_query = tf.repeat(query, values.shape[1], axis=1)
        memory_cell_input = tf.concat([duplicated_query, values], axis=-1)
        # accumulate information via memory cell for each head and concatenate the outputs together
        accumulated_input = tf.concat([memory_cell(memory_cell_input) for memory_cell in self.memory_layer], axis=-1)
        # merge outputs from all heads to size dim via a dense layer and add a dimension for later concatenation
        return tf.expand_dims(self.dense_layer(accumulated_input), axis=1)

    def call(self, inputs, **kwargs):
        # split inputs tuple to the arguments
        queries, values = inputs
        # compute an accumulated representation for all queries
        return tf.concat([self.compute_accumulated_representation(queries[:, query_index:query_index + 1, :], values) for query_index in range(queries.shape[1])], axis=1)


class MemoryAccumulation(tf.keras.layers.Layer):
    def __init__(self, dim, heads):
        super().__init__()
        # create the memory layer that accumulates information
        self.memory_layer = MemoryLayer(dim, heads)

    def call(self, inputs, **kwargs):
        # split inputs tuple to the arguments
        queries, _, values, _ = inputs
        # compute an accumulated representation using the memory layer, no attention weights present
        return self.memory_layer((queries, values)), None
