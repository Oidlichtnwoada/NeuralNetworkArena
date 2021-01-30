"""
code taken from https://github.com/jingli9111/EUNN-tensorflow/blob/master/eunn.py (heavily modified)
"""

import math

import numpy as np
import tensorflow as tf

import experiments.models.model_factory as model_factory


def modrelu(inputs, bias, cplex=True):
    if cplex:
        norm = tf.abs(inputs) + 0.01
        biased_norm = norm + bias
        magnitude = tf.cast(tf.nn.relu(biased_norm), tf.complex64)
        phase = inputs / tf.cast(norm, tf.complex64)
    else:
        norm = tf.abs(inputs) + 0.01
        biased_norm = norm + bias
        magnitude = tf.nn.relu(biased_norm)
        phase = tf.sign(inputs)
    return phase * magnitude


def generate_index_tunable(s, L):
    ind1 = list(range(s))
    ind2 = list(range(s))
    for i in range(s):
        if i % 2 == 1:
            ind1[i] = ind1[i] - 1
            if i == s - 1:
                continue
            else:
                ind2[i] = ind2[i] + 1
        else:
            ind1[i] = ind1[i] + 1
            if i == 0:
                continue
            else:
                ind2[i] = ind2[i] - 1
    ind_exe = [ind1, ind2] * int(L / 2)
    ind3 = []
    ind4 = []
    for i in range(int(s / 2)):
        ind3.append(i)
        ind3.append(i + int(s / 2))
    ind4.append(0)
    for i in range(int(s / 2) - 1):
        ind4.append(i + 1)
        ind4.append(i + int(s / 2))
    ind4.append(s - 1)
    ind_param = [ind3, ind4]
    return ind_exe, ind_param


def ind_s(k):
    if k == 0:
        return np.array([[1, 0]])
    else:
        temp = np.array(range(2 ** k))
        list0 = [np.append(temp + 2 ** k, temp)]
        list1 = ind_s(k - 1)
        for index in range(k):
            list0.append(np.append(list1[index], list1[index] + 2 ** k))
        return list0


def generate_index_fft(s):
    t = ind_s(int(math.log(s / 2, 2)))
    ind_exe = []
    for i in range(int(math.log(s, 2))):
        ind_exe.append(tf.constant(t[i]))
    ind_param = []
    for i in range(int(math.log(s, 2))):
        ind = np.array([])
        for j in range(2 ** i):
            ind = np.append(ind, np.array(range(0, s, 2 ** i)) + j).astype(np.int32)
        ind_param.append(tf.constant(ind))
    return ind_exe, ind_param


@tf.keras.utils.register_keras_serializable()
class EUNNCell(tf.keras.layers.AbstractRNNCell):
    def __init__(self,
                 num_units,
                 capacity=4,
                 fft=False,
                 cplex=True,
                 **kwargs):
        super().__init__(**kwargs)
        self._num_units = num_units
        self._activation = modrelu
        self._capacity = capacity
        self._fft = fft
        self._cplex = cplex
        if self._capacity > self._num_units:
            raise ValueError("Do not set capacity larger than hidden size, it is redundant")
        if self._fft:
            if math.log(self._num_units, 2) % 1 != 0:
                raise ValueError("FFT style only supports power of 2 of hidden size")
        else:
            if self._num_units % 2 != 0:
                raise ValueError("Tunable style only supports even number of hidden size")
            if self._capacity % 2 != 0:
                raise ValueError("Tunable style only supports even number of capacity")
        self.phase_init = tf.random_uniform_initializer(-3.14, 3.14)
        if self._fft:
            self._capacity = int(math.log(self._num_units, 2))
            self.theta, self.phi, self.omega = self.create_fft_weights()
        else:
            self.capacity_A, self.capacity_B, self.theta_A, self.theta_B, self.phi_A, self.phi_B, self.omega = self.create_tunable_weights()
        self.bias = self.add_weight("bias", [self._num_units], initializer=tf.constant_initializer())
        if self._cplex:
            self.U_re, self.U_im = None, None
        else:
            self.U = None

    def build(self, input_shape):
        inputs_size = model_factory.get_concat_input_shape(input_shape)
        input_matrix_init = tf.random_uniform_initializer(-0.01, 0.01)
        if self._cplex:
            self.U_re = self.add_weight("U_re", [inputs_size, self._num_units], initializer=input_matrix_init)
            self.U_im = self.add_weight("U_im", [inputs_size, self._num_units], initializer=input_matrix_init)
        else:
            self.U = self.add_weight("U", [inputs_size, self._num_units], initializer=input_matrix_init)

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if self._cplex:
            dtype = tf.complex64
        else:
            dtype = tf.float32
        return tf.zeros((batch_size, self.state_size), dtype)

    def create_fft_weights(self):
        theta = self.add_weight("theta", [self._capacity, self._num_units // 2], initializer=self.phase_init)
        if self._cplex:
            phi = self.add_weight("phi", [self._capacity, self._num_units // 2], initializer=self.phase_init)
            omega = self.add_weight("omega", [self._num_units], initializer=self.phase_init)
        else:
            phi, omega = None, None
        return theta, phi, omega

    def create_fft_matrices(self):
        cos_theta = tf.cos(self.theta)
        sin_theta = tf.sin(self.theta)
        if self._cplex:
            cos_phi = tf.cos(self.phi)
            sin_phi = tf.sin(self.phi)
            cos_list_re = tf.concat([cos_theta, cos_theta * cos_phi], axis=1)
            cos_list_im = tf.concat([tf.zeros_like(self.theta), cos_theta * sin_phi], axis=1)
            sin_list_re = tf.concat([sin_theta, - sin_theta * cos_phi], axis=1)
            sin_list_im = tf.concat([tf.zeros_like(self.theta), - sin_theta * sin_phi], axis=1)
            cos_list = tf.complex(cos_list_re, cos_list_im)
            sin_list = tf.complex(sin_list_re, sin_list_im)
        else:
            cos_list = tf.concat([cos_theta, cos_theta], axis=1)
            sin_list = tf.concat([sin_theta, -sin_theta], axis=1)
        ind_exe, index_fft = generate_index_fft(self._num_units)
        v1 = tf.stack([tf.gather(cos_list[i, :], index_fft[i]) for i in range(self._capacity)])
        v2 = tf.stack([tf.gather(sin_list[i, :], index_fft[i]) for i in range(self._capacity)])
        if self._cplex:
            D = tf.complex(tf.cos(self.omega), tf.sin(self.omega))
        else:
            D = None
        diag = D
        return v1, v2, ind_exe, diag

    def create_tunable_weights(self):
        capacity_A = int(self._capacity // 2)
        capacity_B = self._capacity - capacity_A
        theta_A = self.add_weight("theta_A", [capacity_A, self._num_units // 2], initializer=self.phase_init)
        theta_B = self.add_weight("theta_B", [capacity_B, self._num_units // 2 - 1], initializer=self.phase_init)
        if self._cplex:
            phi_A = self.add_weight("phi_A", [capacity_A, self._num_units // 2], initializer=self.phase_init)
            phi_B = self.add_weight("phi_B", [capacity_B, self._num_units // 2 - 1], initializer=self.phase_init)
            omega = self.add_weight("omega", [self._num_units], initializer=self.phase_init)
        else:
            phi_A, phi_B, omega = None, None, None
        return capacity_A, capacity_B, theta_A, theta_B, phi_A, phi_B, omega

    def create_tunable_matrices(self):
        cos_theta_A = tf.cos(self.theta_A)
        sin_theta_A = tf.sin(self.theta_A)
        if self._cplex:
            cos_phi_A = tf.cos(self.phi_A)
            sin_phi_A = tf.sin(self.phi_A)
            cos_list_A_re = tf.concat([cos_theta_A, cos_theta_A * cos_phi_A], axis=1)
            cos_list_A_im = tf.concat([tf.zeros_like(self.theta_A), cos_theta_A * sin_phi_A], axis=1)
            sin_list_A_re = tf.concat([sin_theta_A, - sin_theta_A * cos_phi_A], axis=1)
            sin_list_A_im = tf.concat([tf.zeros_like(self.theta_A), - sin_theta_A * sin_phi_A], axis=1)
            cos_list_A = tf.complex(cos_list_A_re, cos_list_A_im)
            sin_list_A = tf.complex(sin_list_A_re, sin_list_A_im)
        else:
            cos_list_A = tf.concat([cos_theta_A, cos_theta_A], axis=1)
            sin_list_A = tf.concat([sin_theta_A, -sin_theta_A], axis=1)
        cos_theta_B = tf.cos(self.theta_B)
        sin_theta_B = tf.sin(self.theta_B)
        if self._cplex:
            cos_phi_B = tf.cos(self.phi_B)
            sin_phi_B = tf.sin(self.phi_B)
            cos_list_B_re = tf.concat([tf.ones([self.capacity_B, 1]), cos_theta_B, cos_theta_B * cos_phi_B, tf.ones([self.capacity_B, 1])], axis=1)
            cos_list_B_im = tf.concat([tf.zeros([self.capacity_B, 1]), tf.zeros_like(self.theta_B), cos_theta_B * sin_phi_B, tf.zeros([self.capacity_B, 1])], axis=1)
            sin_list_B_re = tf.concat([tf.zeros([self.capacity_B, 1]), sin_theta_B, -sin_theta_B * cos_phi_B, tf.zeros([self.capacity_B, 1])], axis=1)
            sin_list_B_im = tf.concat([tf.zeros([self.capacity_B, 1]), tf.zeros_like(self.theta_B), -sin_theta_B * sin_phi_B, tf.zeros([self.capacity_B, 1])], axis=1)
            cos_list_B = tf.complex(cos_list_B_re, cos_list_B_im)
            sin_list_B = tf.complex(sin_list_B_re, sin_list_B_im)
        else:
            cos_list_B = tf.concat([tf.ones([self.capacity_B, 1]), cos_theta_B, cos_theta_B, tf.ones([self.capacity_B, 1])], axis=1)
            sin_list_B = tf.concat([tf.zeros([self.capacity_B, 1]), sin_theta_B, -sin_theta_B, tf.zeros([self.capacity_B, 1])], axis=1)
        ind_exe, [index_A, index_B] = generate_index_tunable(self._num_units, self._capacity)
        diag_list_A = tf.gather(cos_list_A, index_A, axis=1)
        off_list_A = tf.gather(sin_list_A, index_A, axis=1)
        diag_list_B = tf.gather(cos_list_B, index_B, axis=1)
        off_list_B = tf.gather(sin_list_B, index_B, axis=1)
        v1 = tf.reshape(tf.concat([diag_list_A, diag_list_B], axis=1), [self._capacity, self._num_units])
        v2 = tf.reshape(tf.concat([off_list_A, off_list_B], axis=1), [self._capacity, self._num_units])
        if self._cplex:
            D = tf.complex(tf.cos(self.omega), tf.sin(self.omega))
        else:
            D = None
        diag = D
        return v1, v2, ind_exe, diag

    def loop(self, h, v1, v2, ind, _diag):
        for i in range(self._capacity):
            diag = h * v1[i, :]
            off = h * v2[i, :]
            h = diag + tf.gather(off, ind[i], axis=1)
        if _diag is not None:
            h = h * _diag
        return h

    def call(self, inputs, state):
        inputs = model_factory.get_concat_inputs(inputs)
        if self._cplex:
            inputs_re = tf.matmul(inputs, self.U_re)
            inputs_im = tf.matmul(inputs, self.U_im)
            inputs = tf.complex(inputs_re, inputs_im)
        else:
            inputs = tf.matmul(inputs, self.U)
        if self._fft:
            v1, v2, ind, diag = self.create_fft_matrices()
        else:
            v1, v2, ind, diag = self.create_tunable_matrices()
        state = self.loop(state[0], v1, v2, ind, diag)
        output = self._activation((inputs + state), self.bias, self._cplex)
        return output, (output,)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_units': self._num_units,
            'capacity': self._capacity,
            'fft': self._fft,
            'cplex': self._cplex
        })
        return config
