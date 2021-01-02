import numpy as np

import experiments.benchmarks.benchmark as benchmark


class XorBenchmark(benchmark.Benchmark):
    def __init__(self):
        super().__init__('memory', True, False,
                         (('--sequence_length', 4, int),
                          ('--sample_amount', 100_000, int),
                          ('--loss_name', 'SparseCategoricalCrossentropy', str),
                          ('--loss_config', {'from_logits': True}, dict),
                          ('--metric_name', 'SparseCategoricalAccuracy', str)))

    def get_data(self):
        sequence_length = self.args.sequence_length
        sample_amount = self.args.sample_amount
        shape = (sample_amount, sequence_length, 1)
        input_data = np.random.randint(2, size=shape)
        time_data = np.ones_like(input_data)
        output_data = np.sum(input_data, -2) % 2
        return np.stack((input_data, time_data)), np.stack((output_data,)), 2


XorBenchmark()
