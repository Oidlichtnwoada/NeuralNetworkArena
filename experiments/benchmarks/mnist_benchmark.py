import numpy as np
import tensorflow as tf

import experiments.benchmarks.benchmark as benchmark


class MnistBenchmark(benchmark.Benchmark):
    def __init__(self):
        super().__init__('mnist',
                         (('--max_samples', 70_000, int),
                          ('--loss_name', 'SparseCategoricalCrossentropy', str),
                          ('--loss_config', {'from_logits': True}, dict),
                          ('--metric_name', 'SparseCategoricalAccuracy', str)))

    def get_data_and_output_size(self):
        max_samples = self.args.max_samples
        data = tf.keras.datasets.mnist.load_data()
        input_data = np.reshape(np.concatenate((data[0][0], data[1][0])), (-1, 784))[..., np.newaxis]
        time_data = np.ones_like(input_data)
        output_data = np.concatenate((data[0][1], data[1][1]))[..., np.newaxis]
        return (input_data[:max_samples], time_data[:max_samples]), (output_data[:max_samples],), 10


MnistBenchmark()
