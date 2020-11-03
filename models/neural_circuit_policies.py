import numpy as np
import tensorflow as tf


def sigmoid(v_pre, mu, sigma):
    v_pre = tf.expand_dims(v_pre, axis=-1)
    mus = v_pre - mu
    x = sigma * mus
    return tf.nn.sigmoid(x)


class Wiring:
    def __init__(self, units):
        self.units = units
        self.adjacency_matrix = np.zeros([units, units], dtype=np.int32)
        self.input_dim = None
        self.output_dim = None

    def build(self, input_shape):
        input_dim = int(input_shape[1])
        if self.input_dim is not None and self.input_dim != input_dim:
            raise ValueError(
                "Conflicting input dimensions provided. set_input_dim() was called with {} but actual input has dimension {}".format(
                    self.input_dim, input_dim
                )
            )
        if self.input_dim is None:
            self.set_input_dim(input_dim)

    def erev_initializer(self, shape, dtype=None):
        return tf.convert_to_tensor(self.adjacency_matrix, dtype=tf.float32)

    def sensory_erev_initializer(self, shape, dtype=None):
        return tf.convert_to_tensor(self.sensory_adjacency_matrix, dtype=tf.float32)

    def set_input_dim(self, input_dim):
        self.input_dim = input_dim
        self.sensory_adjacency_matrix = np.zeros(
            [input_dim, self.units], dtype=np.int32
        )

    def set_output_dim(self, output_dim):
        self.output_dim = output_dim

    def get_type_of_neuron(self, neuron_id):
        raise NotImplementedError()

    def add_synapse(self, src, dest, polarity):
        if src < 0 or src >= self.units:
            raise ValueError(
                "Cannot add synapse originating in {} if cell has only {} units".format(
                    src, self.units
                )
            )
        if dest < 0 or dest >= self.units:
            raise ValueError(
                "Cannot add synapse feeding into {} if cell has only {} units".format(
                    dest, self.units
                )
            )
        if polarity not in [-1, 1]:
            raise ValueError(
                "Cannot add synapse with polarity {} (expected -1 or +1)".format(
                    polarity
                )
            )
        self.adjacency_matrix[src, dest] = polarity

    def add_sensory_synapse(self, src, dest, polarity):
        if self.input_dim is None:
            raise ValueError(
                "Cannot add sensory synapses before build() has been called!"
            )
        if src < 0 or src >= self.input_dim:
            raise ValueError(
                "Cannot add sensory synapse originating in {} if input has only {} features".format(
                    src, self.input_dim
                )
            )
        if dest < 0 or dest >= self.units:
            raise ValueError(
                "Cannot add synapse feeding into {} if cell has only {} units".format(
                    dest, self.units
                )
            )
        if polarity not in [-1, 1]:
            raise ValueError(
                "Cannot add synapse with polarity {} (expected -1 or +1)".format(
                    polarity
                )
            )
        self.sensory_adjacency_matrix[src, dest] = polarity


class NCP(Wiring):
    def __init__(
            self,
            inter_neurons,
            command_neurons,
            motor_neurons,
            sensory_fanout,
            inter_fanout,
            recurrent_command_synapses,
            motor_fanin,
            seed=22222,
    ):

        super(NCP, self).__init__(inter_neurons + command_neurons + motor_neurons)
        self.set_output_dim(motor_neurons)
        self.rng = np.random.RandomState(seed)
        self.num_inter_neurons = inter_neurons
        self.num_command_neurons = command_neurons
        self.num_motor_neurons = motor_neurons
        self.sensory_fanout = sensory_fanout
        self.inter_fanout = inter_fanout
        self.recurrent_command_synapses = recurrent_command_synapses
        self.motor_fanin = motor_fanin

        # Neuron IDs: [0..motor ... command ... inter]
        self.motor_neurons = list(range(0, self.num_motor_neurons))
        self.command_neurons = list(
            range(
                self.num_motor_neurons,
                self.num_motor_neurons + self.num_command_neurons,
            )
        )
        self.inter_neurons = list(
            range(
                self.num_motor_neurons + self.num_command_neurons,
                self.num_motor_neurons
                + self.num_command_neurons
                + self.num_inter_neurons,
            )
        )

        if self.motor_fanin > self.num_command_neurons:
            raise ValueError(
                "Error: Motor fanin parameter is {} but there are only {} command neurons".format(
                    self.motor_fanin, self.num_command_neurons
                )
            )
        if self.sensory_fanout > self.num_inter_neurons:
            raise ValueError(
                "Error: Sensory fanout parameter is {} but there are only {} inter neurons".format(
                    self.sensory_fanout, self.num_inter_neurons
                )
            )
        if self.inter_fanout > self.num_command_neurons:
            raise ValueError(
                "Error:: Inter fanout parameter is {} but there are only {} command neurons".format(
                    self.inter_fanout, self.num_command_neurons
                )
            )

    def get_type_of_neuron(self, neuron_id):
        if neuron_id < self.num_motor_neurons:
            return "motor"
        if neuron_id < self.num_motor_neurons + self.num_command_neurons:
            return "command"
        return "inter"

    def build_sensory_to_inter_layer(self):
        unreachable_inter_neurons = [x for x in self.inter_neurons]
        # Randomly connects each sensory neuron to exactly _sensory_fanout number of interneurons
        for src in self.sensory_neurons:
            for dest in self.rng.choice(
                    self.inter_neurons, size=self.sensory_fanout, replace=False
            ):
                if dest in unreachable_inter_neurons:
                    unreachable_inter_neurons.remove(dest)
                polarity = self.rng.choice([-1, 1])
                self.add_sensory_synapse(src, dest, polarity)

        # If it happens that some interneurons are not connected, connect them now
        mean_inter_neuron_fanin = int(
            self.num_sensory_neurons * self.sensory_fanout / self.num_inter_neurons
        )
        # Connect "forgotten" inter neuron by at least 1 and at most all sensory neuron
        mean_inter_neuron_fanin = np.clip(
            mean_inter_neuron_fanin, 1, self.num_sensory_neurons
        )
        for dest in unreachable_inter_neurons:
            for src in self.rng.choice(
                    self.sensory_neurons, size=mean_inter_neuron_fanin, replace=False
            ):
                polarity = self.rng.choice([-1, 1])
                self.add_sensory_synapse(src, dest, polarity)

    def build_inter_to_command_layer(self):
        # Randomly connect interneurons to command neurons
        unreachable_command_neurons = [x for x in self.command_neurons]
        for src in self.inter_neurons:
            for dest in self.rng.choice(
                    self.command_neurons, size=self.inter_fanout, replace=False
            ):
                if dest in unreachable_command_neurons:
                    unreachable_command_neurons.remove(dest)
                polarity = self.rng.choice([-1, 1])
                self.add_synapse(src, dest, polarity)

        # If it happens that some command neurons are not connected, connect them now
        mean_command_neurons_fanin = int(
            self.num_inter_neurons * self.inter_fanout / self.num_command_neurons
        )
        # Connect "forgotten" command neuron by at least 1 and at most all inter neuron
        mean_command_neurons_fanin = np.clip(
            mean_command_neurons_fanin, 1, self.num_command_neurons
        )
        for dest in unreachable_command_neurons:
            for src in self.rng.choice(
                    self.inter_neurons, size=mean_command_neurons_fanin, replace=False
            ):
                polarity = self.rng.choice([-1, 1])
                self.add_synapse(src, dest, polarity)

    def build_recurrent_command_layer(self):
        # Add recurrency in command neurons
        for i in range(self.recurrent_command_synapses):
            src = self.rng.choice(self.command_neurons)
            dest = self.rng.choice(self.command_neurons)
            polarity = self.rng.choice([-1, 1])
            self.add_synapse(src, dest, polarity)

    def build_command__to_motor_layer(self):
        # Randomly connect command neurons to motor neurons
        unreachable_command_neurons = [x for x in self.command_neurons]
        for dest in self.motor_neurons:
            for src in self.rng.choice(
                    self.command_neurons, size=self.motor_fanin, replace=False
            ):
                if src in unreachable_command_neurons:
                    unreachable_command_neurons.remove(src)
                polarity = self.rng.choice([-1, 1])
                self.add_synapse(src, dest, polarity)

        # If it happens that some commandneurons are not connected, connect them now
        mean_command_fanout = int(
            self.num_motor_neurons * self.motor_fanin / self.num_command_neurons
        )
        # Connect "forgotten" command neuron to at least 1 and at most all motor neuron
        mean_command_fanout = np.clip(mean_command_fanout, 1, self.num_motor_neurons)
        for src in unreachable_command_neurons:
            for dest in self.rng.choice(
                    self.motor_neurons, size=mean_command_fanout, replace=False
            ):
                polarity = self.rng.choice([-1, 1])
                self.add_synapse(src, dest, polarity)

    def build(self, input_shape):
        super().build(input_shape)
        self.num_sensory_neurons = self.input_dim
        self.sensory_neurons = list(range(0, self.num_sensory_neurons))

        self.build_sensory_to_inter_layer()
        self.build_inter_to_command_layer()
        self.build_recurrent_command_layer()
        self.build_command__to_motor_layer()


class LTCCell(tf.keras.layers.Layer):
    def __init__(
            self,
            wiring,
            input_mapping="affine",
            output_mapping="affine",
            ode_unfolds=6,
            epsilon=1e-8,
            initialization_ranges=None,
            **kwargs
    ):

        self.init_ranges = {
            "gleak": (0.001, 1.0),
            "vleak": (-0.2, 0.2),
            "cm": (0.4, 0.6),
            "w": (0.001, 1.0),
            "sigma": (3, 8),
            "mu": (0.3, 0.8),
            "sensory_w": (0.001, 1.0),
            "sensory_sigma": (3, 8),
            "sensory_mu": (0.3, 0.8),
        }
        if initialization_ranges is not None:
            for k, v in initialization_ranges.items():
                if k not in self.init_ranges.keys():
                    raise ValueError(
                        "Unknown parameter '{}' in initialization range dictionary! (Expected only {})".format(
                            k, str(list(self.init_range.keys()))
                        )
                    )
                if k in ["gleak", "cm", "w", "sensory_w"] and v[0] < 0:
                    raise ValueError(
                        "Initialization range of parameter '{}' must be non-negative!".format(
                            k
                        )
                    )
                if v[0] > v[1]:
                    raise ValueError(
                        "Initialization range of parameter '{}' is not a valid range".format(
                            k
                        )
                    )
                self.init_ranges[k] = v

        self.wiring = wiring
        self.input_mapping = input_mapping
        self.output_mapping = output_mapping
        self.ode_unfolds = ode_unfolds
        self.epsilon = epsilon
        super(LTCCell, self).__init__(name="wormnet", **kwargs)

    @property
    def state_size(self):
        return self.wiring.units

    @property
    def sensory_size(self):
        return self.wiring.input_dim

    @property
    def motor_size(self):
        return self.wiring.output_dim

    @property
    def synapse_count(self):
        return np.sum(np.abs(self.wiring.adjacency_matrix))

    @property
    def sensory_synapse_count(self):
        return np.sum(np.abs(self.wiring.adjacency_matrix))

    def get_initializer(self, param_name):
        minval, maxval = self.init_ranges[param_name]
        if minval == maxval:
            return tf.keras.initializers.Constant(minval)
        else:
            return tf.keras.initializers.RandomUniform(minval, maxval)

    def build(self, input_shape):
        if isinstance(input_shape, (tuple, list)):
            input_shape = input_shape[0]

        self.wiring.build(input_shape)

        self.params = {
            "gleak": self.add_weight(
                name="gleak",
                shape=(self.state_size,),
                dtype=tf.float32,
                constraint=tf.keras.constraints.NonNeg(),
                initializer=self.get_initializer("gleak"),
            ), "vleak": self.add_weight(
                name="vleak",
                shape=(self.state_size,),
                dtype=tf.float32,
                initializer=self.get_initializer("vleak"),
            ), "cm": self.add_weight(
                name="cm",
                shape=(self.state_size,),
                dtype=tf.float32,
                constraint=tf.keras.constraints.NonNeg(),
                initializer=self.get_initializer("cm"),
            ), "sigma": self.add_weight(
                name="sigma",
                shape=(self.state_size, self.state_size),
                dtype=tf.float32,
                initializer=self.get_initializer("sigma"),
            ), "mu": self.add_weight(
                name="mu",
                shape=(self.state_size, self.state_size),
                dtype=tf.float32,
                initializer=self.get_initializer("mu"),
            ), "w": self.add_weight(
                name="w",
                shape=(self.state_size, self.state_size),
                dtype=tf.float32,
                constraint=tf.keras.constraints.NonNeg(),
                initializer=self.get_initializer("w"),
            ), "erev": self.add_weight(
                name="erev",
                shape=(self.state_size, self.state_size),
                dtype=tf.float32,
                initializer=self.wiring.erev_initializer,
            ), "sensory_sigma": self.add_weight(
                name="sensory_sigma",
                shape=(self.sensory_size, self.state_size),
                dtype=tf.float32,
                initializer=self.get_initializer("sensory_sigma"),
            ), "sensory_mu": self.add_weight(
                name="sensory_mu",
                shape=(self.sensory_size, self.state_size),
                dtype=tf.float32,
                initializer=self.get_initializer("sensory_mu"),
            ), "sensory_w": self.add_weight(
                name="sensory_w",
                shape=(self.sensory_size, self.state_size),
                dtype=tf.float32,
                constraint=tf.keras.constraints.NonNeg(),
                initializer=self.get_initializer("sensory_w"),
            ), "sensory_erev": self.add_weight(
                name="sensory_erev",
                shape=(self.sensory_size, self.state_size),
                dtype=tf.float32,
                initializer=self.wiring.sensory_erev_initializer,
            ), "sparsity_mask": tf.constant(
                np.abs(self.wiring.adjacency_matrix), dtype=tf.float32
            ), "sensory_sparsity_mask": tf.constant(
                np.abs(self.wiring.sensory_adjacency_matrix), dtype=tf.float32
            )}

        if self.input_mapping in ["affine", "linear"]:
            self.params["input_w"] = self.add_weight(
                name="input_w",
                shape=(self.sensory_size,),
                dtype=tf.float32,
                initializer=tf.keras.initializers.Constant(1),
            )
        if self.input_mapping == "affine":
            self.params["input_b"] = self.add_weight(
                name="input_b",
                shape=(self.sensory_size,),
                dtype=tf.float32,
                initializer=tf.keras.initializers.Constant(0),
            )

        if self.output_mapping in ["affine", "linear"]:
            self.params["output_w"] = self.add_weight(
                name="output_w",
                shape=(self.motor_size,),
                dtype=tf.float32,
                initializer=tf.keras.initializers.Constant(1),
            )
        if self.output_mapping == "affine":
            self.params["output_b"] = self.add_weight(
                name="output_b",
                shape=(self.motor_size,),
                dtype=tf.float32,
                initializer=tf.keras.initializers.Constant(0),
            )
        self.built = True

    def ode_solver(self, inputs, state, elapsed_time):
        v_pre = state

        # We can pre-compute the effects of the sensory neurons here
        sensory_w_activation = self.params["sensory_w"] * sigmoid(
            inputs, self.params["sensory_mu"], self.params["sensory_sigma"]
        )
        sensory_w_activation *= self.params["sensory_sparsity_mask"]

        sensory_rev_activation = sensory_w_activation * self.params["sensory_erev"]

        # Reduce over dimension 1 (=source sensory neurons)
        w_numerator_sensory = tf.reduce_sum(sensory_rev_activation, axis=1)
        w_denominator_sensory = tf.reduce_sum(sensory_w_activation, axis=1)

        # cm/t is loop invariant
        cm_t = self.params["cm"] / (elapsed_time / self.ode_unfolds)

        # Unfold the mutliply ODE multiple times into one RNN step
        for t in range(self.ode_unfolds):
            w_activation = self.params["w"] * sigmoid(
                v_pre, self.params["mu"], self.params["sigma"]
            )

            w_activation *= self.params["sparsity_mask"]

            rev_activation = w_activation * self.params["erev"]

            # Reduce over dimension 1 (=source neurons)
            w_numerator = tf.reduce_sum(rev_activation, axis=1) + w_numerator_sensory
            w_denominator = tf.reduce_sum(w_activation, axis=1) + w_denominator_sensory

            numerator = (
                    cm_t * v_pre
                    + self.params["gleak"] * self.params["vleak"]
                    + w_numerator
            )
            denominator = cm_t + self.params["gleak"] + w_denominator

            # Avoid dividing by 0
            v_pre = numerator / (denominator + self.epsilon)

        return v_pre

    def map_inputs(self, inputs):
        if self.input_mapping in ["affine", "linear"]:
            inputs = inputs * self.params["input_w"]
        if self.input_mapping == "affine":
            inputs = inputs + self.params["input_b"]
        return inputs

    def map_outputs(self, state):
        output = state
        if self.motor_size < self.state_size:
            output = output[:, 0: self.motor_size]

        if self.output_mapping in ["affine", "linear"]:
            output = output * self.params["output_w"]
        if self.output_mapping == "affine":
            output = output + self.params["output_b"]
        return output

    def call(self, inputs, states):
        if isinstance(inputs, (tuple, list)):
            # irregularly sampled mode
            inputs, elapsed_time = inputs
        else:
            # regularly sampled mode
            elapsed_time = 1.0
        inputs = self.map_inputs(inputs)

        next_state = self.ode_solver(inputs, states[0], elapsed_time)

        outputs = self.map_outputs(next_state)

        return outputs, [next_state]


class NeuralCircuitPolicies(tf.keras.Model):
    def __init__(self, output_length, inter_neurons, command_neurons, motor_neurons, sensory_fanout, inter_fanout, recurrent_command_synapses, motor_fanin):
        super(NeuralCircuitPolicies, self).__init__()
        # parameters
        self.output_length = output_length
        self.inter_neurons = inter_neurons
        self.command_neurons = command_neurons
        self.motor_neurons = motor_neurons
        self.sensory_fanout = sensory_fanout
        self.inter_fanout = inter_fanout
        self.recurrent_command_synapses = recurrent_command_synapses
        self.motor_fanin = motor_fanin
        # used layers
        self.rnn = tf.keras.layers.RNN(
            LTCCell(NCP(self.inter_neurons, self.command_neurons, self.motor_neurons, self.sensory_fanout, self.inter_fanout, self.recurrent_command_synapses, self.motor_fanin)),
            return_sequences=True)
        self.dense_layer = tf.keras.layers.Dense(self.output_length)

    def call(self, inputs, training=None, mask=None):
        signals, times = inputs
        signals = tf.cast(signals, dtype=tf.float32)
        times = tf.cast(times, dtype=tf.float32)
        rnn_output = self.rnn((signals, times))
        return self.dense_layer(rnn_output)

    def get_config(self):
        pass
