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
        # create excitatory recurrent connections to each neuron
        for src in range(self.units):
            self.add_synapse(src, src, polarity=1)
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
