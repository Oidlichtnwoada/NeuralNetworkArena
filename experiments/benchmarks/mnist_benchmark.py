import numpy as np
import tensorflow as tf

import experiments.benchmarks.benchmark as benchmark


class MnistBenchmark(benchmark.Benchmark):
    def __init__(self):
        super().__init__('mnist', True, False,
                         (('--max_sample_amount', 1_000, int),
                          ('--loss_name', 'SparseCategoricalCrossentropy', str),
                          ('--loss_config', {'from_logits': True}, dict),
                          ('--metric_name', 'SparseCategoricalAccuracy', str)))

    def get_data(self):
        max_sample_amount = self.args.max_sample_amount
        data = tf.keras.datasets.mnist.load_data()
        input_data = np.reshape(np.concatenate((data[0][0], data[1][0])), (-1, 784))[..., np.newaxis]
        output_data = np.concatenate((data[0][1], data[1][1]))[..., np.newaxis]
        time_data = np.ones_like(output_data)
        return (input_data[:max_sample_amount].tolist(), time_data[:max_sample_amount].tolist()), (output_data[:max_sample_amount],), 10


MnistBenchmark()
