# -*- coding: utf-8 -*-
# pylint: disable=W0221
"""
Differentiable Neural Computer model definition.

Reference:
    http://www.nature.com/nature/journal/vaop/ncurrent/full/nature20101.html

Conventions:
    B - batch size
    N - number of slots in memory
    R - number of read heads
    W - size of each memory slot i.e word size
"""

import collections

import tensorflow as tf

import experiments.models.model_factory as model_factory

# -*- coding: utf-8 -*-
"""DNC memory operations and state.

Conventions:
    B - batch size
    N - number of slots in memory
    R - number of read heads
"""

EPSILON = 1e-6


class ContentAddressing:
    """
    Access memory content using cosine similarity.

        Used for: reading, writing
    """

    @staticmethod
    def weighting(memory_matrix, keys, strengths, sharpness_op=tf.math.softplus):
        """Get content-based weighting using cosine similarity. The weighting
        for each memory slot will be high if the key points in the same
        direction as the memory contents at that slot.

        Args:
            memory_matrix (Tensor [B, N, W]): the memory matrix to query
            keys (Tensor [B, W, R]): the keys to query the memory
            strengths (Tensor [B, R]): strengths for each lookup key
            sharpness_op (fn): operation to transform strengths before softmax


        Returns:
            Tensor [B, N, R]: lookup weightings for each key
        """
        memory_normalised = tf.math.l2_normalize(memory_matrix, 2, epsilon=EPSILON)
        keys_normalised = tf.math.l2_normalize(keys, 1, epsilon=EPSILON)
        similarity = tf.matmul(memory_normalised, keys_normalised)
        strengths = tf.expand_dims(sharpness_op(strengths), 1)

        return tf.math.softmax(similarity * strengths, 1)


class TemporalLinkAddressing:
    """
    Access memory content by considering which interactions have happened
    recently in time.

        Used for: reading
    """

    @staticmethod
    def update_precedence_vector(prev_precedence_vector, write_weighting):
        """Return next precedence vector by taking into account the writing
        action that has just happened via `write_weighting`.

        The precedence vector at position `i` denotes the degree to which
        memory location `i` has been recently written.

        Args:
            prev_precedence_vector (Tensor [B, N]): precedence vector from time t-1
            write_weighting (Tensor [B, N]): final weighting used to write at time t

        Returns:
            Tensor [B, N]: precedence vector to use at next time step
        """
        write_strength = tf.reduce_sum(input_tensor=write_weighting, axis=1, keepdims=True)
        updated_precedence_vector = (1 - write_strength) * prev_precedence_vector + write_weighting

        return updated_precedence_vector

    @staticmethod
    def update_link_matrix(prev_link_matrix, prev_precedence_vector, write_weighting):
        """Adjust the link matrix by taking into account the writing
        action that has just happened and the previous precedence vector.

        Link matrix at `L[t,i,j]` describes the degree to which memory location
        `i` was written after location `j` between time `t` and `t+1`.

        Args:
            prev_link_matrix (Tensor [B, N, N]): link matrix from time t-1
            prev_precedence_vector (Tensor [B, N]): precedence vector from time t-1
            write_weighting (Tensor [B, N)): final weighting used to write at time t

        Returns:
            Tensor [B, N, N]: temporal link matrix to use at next time step
        """
        batch_size = prev_link_matrix.shape[0]
        if batch_size is None:
            return prev_link_matrix
        words_num = prev_link_matrix.shape[1]

        write_weighting_i = tf.expand_dims(write_weighting, 2)  # [b x N x 1 ] duplicate columns
        write_weighting_j = tf.expand_dims(write_weighting, 1)  # [b x 1 X N ] duplicate rows
        prev_precedence_vector_j = tf.expand_dims(prev_precedence_vector, 1)  # [b x 1 X N]

        link_matrix = (
                (1 - write_weighting_i - write_weighting_j) * prev_link_matrix
                + (write_weighting_i * prev_precedence_vector_j)
        )
        zero_diagonal = tf.zeros([batch_size, words_num], dtype=link_matrix.dtype)

        return tf.linalg.set_diag(link_matrix, zero_diagonal)

    @staticmethod
    def weightings(link_matrix, prev_read_weightings):
        """Calculate weightings for each read head so they have a preference
        towards directionality.

        Args:
            link_matrix (Tensor [B, N, N])
            prev_read_weightings (Tensor [B, N, R]): read weightings from time t-1

        Returns:
            Tuple(Tensor [B, N, R], Tensor [B, N, R]): temporal weightings for each memory slot
        """
        forward_weighting = tf.matmul(link_matrix, prev_read_weightings)
        backward_weighting = tf.matmul(link_matrix, prev_read_weightings, adjoint_a=True)

        return forward_weighting, backward_weighting


class AllocationAddressing:
    """
    Access memory content by considering which memory slots can be allocated to.
    This is used to provide a differentiable form of dynamic memory allocation
    where slots can only be written to if they are determined to be free.

        Used for: writing
    """

    @staticmethod
    def update_usage_vector(free_gates, prev_read_weightings,
                            prev_write_weighting, prev_usage_vector):
        """Adjust the usage vector based on reads and writes from previous time
        step.

        The usage vector is a helper data structure to aid in the calculation of
        the allocation weighting. `u[t,i]` describes the usage between [0,1]
        inside memory slot `i` at time `t`. Elements of usage vector may add up
        to a maximum of `N`.

        The free gate allows reads to happen over multiple time steps at the same
        location, otherwise we would always say a location is unused immediately
        after a read has occurred.

        Args:
            free_gates (Tensor [B, R]): current free gate
            prev_read_weightings (Tensor [B, N, R]): read weightings from time t-1
            prev_write_weighting (Tensor [B, N]): write weighting from time t-1
            prev_usage_vector (Tensor [B, N]): usage vector from time t-1

        Returns:
            Tensor [B, N]: new usage vector
        """
        with tf.name_scope('allocation_addressing'):
            retention_vector = tf.reduce_prod(
                input_tensor=1 - tf.expand_dims(free_gates, 1) * prev_read_weightings,
                axis=2,
            )
            usage_vector = (
                    (prev_usage_vector + prev_write_weighting
                     - (prev_usage_vector * prev_write_weighting))
                    * retention_vector
            )
            return usage_vector

    @staticmethod
    def batch_unsort(tensor, indices):
        """Permute each batch in a batch first tensor according to tensor
        of indices.
        """
        if indices.shape[0] is None:
            return tensor
        unpacked = tf.unstack(indices)
        indices_inverted = tf.stack(
            [tf.math.invert_permutation(permutation) for permutation in unpacked]
        )

        unpacked = zip(tf.unstack(tensor), tf.unstack(indices_inverted))
        return tf.stack([tf.gather(value, index) for value, index in unpacked])

    @staticmethod
    def weighting(usage_vector):
        """Calculate allocation weighting so we know which memory slots are
        free to be written to. Tells us the degree to which each memory location
        is "allocable".

        Args:
            usage_vector (Tensor [B, N]): newly calculated usage vector at time t

        Returns:
            Tensor [B, N]: allocation weighting for each memory slot
        """
        usage = (1 - EPSILON) * usage_vector + EPSILON
        emptiness = 1 - usage

        words_num = usage_vector.get_shape().as_list()[1]
        emptiness_sorted, free_list = tf.nn.top_k(emptiness, k=words_num)
        usage_sorted = 1 - emptiness_sorted
        allocation_sorted = emptiness_sorted * tf.math.cumprod(usage_sorted, axis=1, exclusive=True)

        return AllocationAddressing.batch_unsort(allocation_sorted, free_list)


class Memory:
    """Differentiable memory for the DNC.

    This module implements a recurrent module interface and tracks memory state
    through time. Performs a write and read operation given the previous state
    and an interface vector defining how to interact with the memory at the
    current time step.

    Note: although this layer behaves similar to an rnn, it has no parameters
    and is actually a deterministic operation:
        (interface, prev_memory_state) -> (read_vectors, new_memory_state)

    Args:
        words_num (int): number of memory slots
        word_size (int): size of each memory slot
        read_heads_num (int): number of read heads to use inside memory
    """

    state = collections.namedtuple(
        "memory_state", [
            'memory_matrix',
            'usage_vector',
            'link_matrix',
            'precedence_vector',
            'write_weighting',
            'read_weightings',
        ]
    )

    def __init__(self, words_num=256, word_size=64, read_heads_num=4):
        self._N = words_num
        self._W = word_size
        self._R = read_heads_num

    def __call__(self, interface, prev_memory_state):
        """Define op for the recurrent module.

        Args:
            interface (namedtuple): parsed interface vector
            prev_memory_state (namedtuple): object containing the memory plus all
                the helper data structures used to interface with the memory

        Returns:
            Tuple:
                read vectors (Tensor [N, R]): read vectors taken out of the memory
                next memory state (namedtuple): new state after write and read
        """
        with tf.name_scope("write"):
            usage, write_weighting, memory_matrix, link_matrix, precedence = Memory.write(
                prev_memory_state,
                interface,
            )

        with tf.name_scope("read"):
            read_weightings, read_vectors = Memory.read(
                memory_matrix,
                prev_memory_state.read_weightings,
                link_matrix,
                interface,
            )
        return read_vectors, Memory.state(
            memory_matrix=memory_matrix,
            usage_vector=usage,
            link_matrix=link_matrix,
            precedence_vector=precedence,
            write_weighting=write_weighting,
            read_weightings=read_weightings,
        )

    @staticmethod
    def read(memory_matrix, prev_read_weightings, link_matrix, interface):
        """Perform read on memory.

        Args:
            memory_matrix (Tensor [B, N, W]): memory matrix after recent write at time t
            prev_read_weightings (Tensor [B, N, R]): read weightings from time t-1
            link_matrix (Tensor [B, N, N]): link matrix after recent write at time t
            interface (namedtuple): parsed interface vector

        Returns:
            Tuple:
                read_weightings (Tensor [B, N, R]): read vectors taken out of the memory
                read_vectors (Tensor [B, W, R]): read vectors taken out of the memory

        """
        with tf.name_scope("content_addressing"):
            lookup_weighting = ContentAddressing.weighting(
                memory_matrix,
                interface.read_keys,
                interface.read_strengths
            )
        with tf.name_scope("temporal_link_addressing"):
            forward_weighting, backward_weighting = TemporalLinkAddressing.weightings(
                link_matrix,
                prev_read_weightings,
            )

        with tf.name_scope("blend_addressing_modes"):
            read_weightings = tf.einsum(
                "bsr,bnrs->bnr",
                interface.read_modes,
                tf.stack([backward_weighting, lookup_weighting, forward_weighting], axis=3)
            )
        read_vectors = tf.matmul(memory_matrix, read_weightings, adjoint_a=True)

        return read_weightings, read_vectors

    @staticmethod
    def write(prev_memory_state, interface):
        """Perform write on memory.

        Args:
            prev_memory_state (namedtuple): memory state from time t-1
            interface (namedtuple): parsed interface vector

        Returns:
            Tuple:
                usage_vector (Tensor [B, N])
                write_weighting (Tensor [B, N])
                memory_matrix (Tensor [B, N, W])
                link_matrix (Tensor [B, N, N])
                precedence_vector (Tensor [B, N])
        """
        m = prev_memory_state
        i = interface

        with tf.name_scope("calculate_weighting"):
            with tf.name_scope("allocation_addressing"):
                usage_vector = AllocationAddressing.update_usage_vector(
                    i.free_gates,
                    m.read_weightings,
                    m.write_weighting,
                    m.usage_vector
                )
                allocation_weighting = AllocationAddressing.weighting(usage_vector)
            with tf.name_scope("content_addressing"):
                lookup_weighting = ContentAddressing.weighting(
                    m.memory_matrix,
                    i.write_key,
                    i.write_strength
                )
            write_weighting = (
                    i.write_gate * (i.allocation_gate * allocation_weighting +
                                    (1 - i.allocation_gate) * tf.squeeze(lookup_weighting))
            )

        with tf.name_scope("erase_and_write"):
            erase = m.memory_matrix * (
                (1 - tf.einsum("bn,bw->bnw", write_weighting, i.erase_vector)))
            write = tf.einsum("bn,bw->bnw", write_weighting, i.write_vector)
            memory_matrix = erase + write

        with tf.name_scope("final_update"):
            link_matrix = TemporalLinkAddressing.update_link_matrix(
                m.link_matrix,
                m.precedence_vector,
                write_weighting
            )
            precedence_vector = TemporalLinkAddressing.update_precedence_vector(
                m.precedence_vector,
                write_weighting
            )

        return usage_vector, write_weighting, memory_matrix, \
               link_matrix, precedence_vector

    @property
    def state_size(self):
        return Memory.state(
            memory_matrix=tf.TensorShape([self._N, self._W]),
            usage_vector=tf.TensorShape([self._N]),
            link_matrix=tf.TensorShape([self._N, self._N]),
            precedence_vector=tf.TensorShape([self._N]),
            write_weighting=tf.TensorShape([self._N]),
            read_weightings=tf.TensorShape([self._N, self._R]),
        )

    def get_initial_state(self, batch_size, dtype=tf.float32):
        return Memory.state(
            memory_matrix=tf.fill([batch_size, self._N, self._W], EPSILON),
            usage_vector=tf.zeros([batch_size, self._N], dtype=dtype),
            link_matrix=tf.zeros([batch_size, self._N, self._N], dtype=dtype),
            precedence_vector=tf.zeros([batch_size, self._N], dtype=dtype),
            write_weighting=tf.fill([batch_size, self._N], EPSILON),
            read_weightings=tf.fill([batch_size, self._N, self._R], EPSILON),
        )


@tf.keras.utils.register_keras_serializable()
class DNC(tf.keras.layers.AbstractRNNCell):
    """DNC recurrent module that connects together the controller and memory.

    Performs a write and read operation against memory given 1) the previous state
    and 2) an interface vector defining how to interact with the memory at the
    current time step.

    Args:
        output_size (int): size of final output dimension for the whole DNC cell at each time step
        controller_units (int): size of hidden state in controller
        memory_size (int): number of slots in external memory
        word_size (int): the width of each memory slot
        num_read_heads (int): number of memory read heads
    """

    state = collections.namedtuple("dnc_state", [
        "memory_state",
        "controller_state",
        "read_vectors",
    ])

    interface = collections.namedtuple("interface", [
        "read_keys",
        "read_strengths",
        "write_key",
        "write_strength",
        "erase_vector",
        "write_vector",
        "free_gates",
        "allocation_gate",
        "write_gate",
        "read_modes",
    ])

    def __init__(self, output_size, controller_units=256, memory_size=256,
                 word_size=64, num_read_heads=4, **kwargs):
        super().__init__(**kwargs)

        self._output_size = output_size
        self._N = memory_size
        self._R = num_read_heads
        self._W = word_size
        self._interface_vector_size = self._R * self._W + 3 * self._W + 5 * self._R + 3
        self._clip = 20.0

        self._controller = tf.keras.layers.LSTMCell(units=controller_units)
        self._controller_to_interface_dense = tf.keras.layers.Dense(
            self._interface_vector_size,
            name='controller_to_interface'
        )
        self._memory = Memory(memory_size, word_size, num_read_heads)
        self._final_output_dense = tf.keras.layers.Dense(self._output_size)

    def _parse_interface_vector(self, interface_vector):
        r, w = self._R, self._W

        sizes = [r * w, r, w, 1, w, w, r, 1, 1, 3 * r]
        fns = collections.OrderedDict([
            ("read_keys", lambda v: tf.reshape(v, (-1, w, r))),
            ("read_strengths", lambda v: 1 + tf.nn.softplus((tf.reshape(v, (-1, r))))),
            ("write_key", lambda v: tf.reshape(v, (-1, w, 1))),
            ("write_strength", lambda v: 1 + tf.nn.softplus((tf.reshape(v, (-1, 1))))),
            ("erase_vector", lambda v: tf.nn.sigmoid(tf.reshape(v, (-1, w)))),
            ("write_vector", lambda v: tf.reshape(v, (-1, w))),
            ("free_gates", lambda v: tf.nn.sigmoid(tf.reshape(v, (-1, r)))),
            ("allocation_gate", lambda v: tf.nn.sigmoid(tf.reshape(v, (-1, 1)))),
            ("write_gate", lambda v: tf.nn.sigmoid(tf.reshape(v, (-1, 1)))),
            ("read_modes", lambda v: tf.nn.softmax(tf.reshape(v, (-1, 3, r)), axis=1)),
        ])
        indices = [[sum(sizes[:i]), sum(sizes[:i + 1])] for i in range(len(sizes))]
        zipped_items = zip(fns.keys(), fns.values(), indices)
        interface = {name: fn(interface_vector[:, i[0]:i[1]]) for name, fn, i in zipped_items}

        return DNC.interface(**interface)

    def _flatten_read_vectors(self, x):
        return tf.reshape(x, (-1, self._W * self._R))

    def call(self, inputs, prev_dnc_state):
        inputs = model_factory.get_concat_inputs(inputs)
        prev_dnc_state = tf.nest.pack_sequence_as(self.state_size_nested, prev_dnc_state)
        with tf.name_scope("inputs_to_controller"):
            read_vectors_flat = self._flatten_read_vectors(prev_dnc_state.read_vectors)
            input_augmented = tf.concat([inputs, read_vectors_flat], 1)
            controller_output, controller_state = self._controller(
                input_augmented,
                prev_dnc_state.controller_state,
            )
            controller_output = tf.clip_by_value(controller_output, -self._clip, self._clip)

        with tf.name_scope("parse_interface"):
            interface = self._controller_to_interface_dense(controller_output)
            interface = self._parse_interface_vector(interface)

        with tf.name_scope("update_memory"):
            read_vectors, memory_state = self._memory(interface, prev_dnc_state.memory_state)
            state = DNC.state(
                memory_state=memory_state,
                controller_state=controller_state,
                read_vectors=read_vectors,
            )

        with tf.name_scope("join_outputs"):
            read_vectors_flat = self._flatten_read_vectors(read_vectors)
            final_output = tf.concat([controller_output, read_vectors_flat], 1)
            final_output = self._final_output_dense(final_output)
            final_output = tf.clip_by_value(final_output, -self._clip, self._clip)

        return final_output, tf.nest.flatten(state)

    @property
    def state_size_nested(self):
        return DNC.state(
            memory_state=self._memory.state_size,
            controller_state=self._controller.state_size,
            read_vectors=tf.TensorShape([self._W, self._R]),
        )

    @property
    def state_size(self):
        return tf.nest.flatten(self.state_size_nested)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=tf.float32):
        del inputs
        initial_state_nested = DNC.state(
            memory_state=self._memory.get_initial_state(batch_size, dtype=dtype),
            controller_state=self._controller.get_initial_state(batch_size=batch_size, dtype=dtype),
            read_vectors=tf.fill([batch_size, self._W, self._R], EPSILON),
        )
        return tf.nest.flatten(initial_state_nested)

    @property
    def output_size(self):
        return self._output_size

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_size': self.output_size,
            'controller_units': self._controller.units,
            'memory_size': self._N,
            'word_size': self._W,
            'num_read_heads': self._R
        })
        return config
