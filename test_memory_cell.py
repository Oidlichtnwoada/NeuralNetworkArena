import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import experiments.models.memory_cell as memory_cell

print(memory_cell)
first = True
second = True
model = tf.keras.models.load_model('benchmark_logs/run_1/saved_models/cell/memory_cell/')
first_neuron_input_data = np.zeros((1, 384, 1))
second_neuron_input_data = np.zeros((1, 384, 1))
for i in range(384):
    if random.random() > 0.96:
        if random.random() < 0.5:
            first_neuron_input_data[0, i, 0] = 1
        else:
            second_neuron_input_data[0, i, 0] = 1
input_data = np.concatenate((first_neuron_input_data, second_neuron_input_data), axis=-1)
output_data = model((input_data, np.ones((0, 384, 0))))
for i in range(384):
    if first_neuron_input_data[0, i, 0] == 1:
        label = 'first neuron sparse activation (vertical)' if first else None
        first = False
        plt.vlines(i, -0.1, 1.1, color='indianred', label=label)
    if second_neuron_input_data[0, i, 0] == 1:
        label = 'second neuron sparse activation (vertical)' if second else None
        second = False
        plt.vlines(i, -0.1, 1.1, color='cornflowerblue', label=label)
plt.plot(output_data[0, :, 0], label='first neuron potential', color='darkred')
plt.plot(output_data[0, :, 1], label='second neuron potential', color='darkblue')
plt.legend()
plt.title('memory cell operation')
plt.xlabel('time steps')
plt.ylabel('potential')
plt.savefig('memory_cell_operation.pdf')
plt.close()
