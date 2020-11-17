import kerasncp as ncp
import tensorflow as tf


class MemoryCellWiring(ncp.wirings.Wiring):
    def __init__(self, dim):
        # input_dim = output_dim = dim
        self.dim = dim
        # allocate data structures for dim neuron pairs
        super().__init__(units=self.dim * 2)
        # the output dimension is half the state dimension of all neuron pairs
        self.set_output_dim(output_dim=self.dim)
        # create inhibitory recurrent connections to each neuron
        for src in range(self.units):
            self.add_synapse(src, src, -1)
        # create inhibitory connections between each neuron pair
        for src in range(self.units // 2):
            self.add_synapse(src, src + self.dim, -1)
            self.add_synapse(src + self.dim, src, -1)

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
        # save the dimension and the number of heads
        self.dim = dim
        self.heads = heads
        # create a memory layer out of heads memory cells
        self.memory_layer = [MemoryCell(dim) for _ in range(self.heads)]
        # create a dense layer to merge all heads
        self.dense_layer = tf.keras.layers.Dense(self.dim)

    def call(self, inputs, **kwargs):
        # save the amount of inputs
        input_amount = inputs.shape[1]
        memory_layer_output = None
        # compute an accumulated representation for all inputs
        for input_index in range(input_amount):
            accumulated_input = None
            # the inputs are accumulated such that the last input to the RNN is the input at input_index
            shifted_inputs = tf.roll(inputs, input_amount - input_index - 1, axis=1)
            for head in range(self.heads):
                # accumulate input via memory cell at index head
                head_output = self.memory_layer[head](shifted_inputs)
                # concatenate output to existing output
                if accumulated_input is None:
                    accumulated_input = head_output
                else:
                    accumulated_input = tf.concat([accumulated_input, head_output], axis=-1)
            # merge outputs from all heads
            accumulated_input = tf.expand_dims(self.dense_layer(accumulated_input), axis=1)
            # concatenate all accumulated inputs together
            if memory_layer_output is None:
                memory_layer_output = accumulated_input
            else:
                memory_layer_output = tf.concat([memory_layer_output, accumulated_input], axis=1)
        # the output contains all accumulated inputs
        return memory_layer_output
