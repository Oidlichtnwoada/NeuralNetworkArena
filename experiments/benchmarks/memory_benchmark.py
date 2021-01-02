import numpy as np

import experiments.benchmarks.benchmark as benchmark


class MemoryBenchmark(benchmark.Benchmark):
    def __init__(self):
        super().__init__('memory', True,
                         (('--memory_length', 100, int),
                          ('--sequence_length', 10, int),
                          ('--category_amount', 10, int),
                          ('--sample_amount', 10_000, int),
                          ('--loss_name', 'SparseCategoricalCrossentropy', str),
                          ('--loss_config', {'from_logits': True}, dict),
                          ('--metric_name', 'SparseCategoricalAccuracy', str)))

    def get_data(self):
        memory_length = self.args.memory_length
        sequence_length = self.args.sequence_length
        category_amount = self.args.category_amount
        sample_amount = self.args.sample_amount
        memory_sequence = np.random.randint(low=0, high=category_amount - 2, size=(sample_amount, sequence_length, 1))
        first_blank_sequence = (category_amount - 2) * np.ones((sample_amount, memory_length - 1, 1))
        marker_sequence = (category_amount - 1) * np.ones((sample_amount, 1, 1))
        second_blank_sequence = (category_amount - 2) * np.ones((sample_amount, sequence_length, 1))
        input_sequence = np.concatenate((memory_sequence, first_blank_sequence, marker_sequence, second_blank_sequence), 1)
        time_sequence = np.ones_like(input_sequence)
        third_blank_sequence = (category_amount - 2) * np.ones((sample_amount, memory_length + sequence_length, 1))
        output_sequence = np.concatenate((third_blank_sequence, memory_sequence), 1)
        return np.stack((input_sequence, time_sequence)), np.stack((output_sequence,)), self.args.category_amount


MemoryBenchmark()
