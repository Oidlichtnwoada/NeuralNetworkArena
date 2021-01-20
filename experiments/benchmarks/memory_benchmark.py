import numpy as np

import experiments.benchmarks.benchmark as benchmark


class MemoryBenchmark(benchmark.Benchmark):
    def __init__(self):
        super().__init__('memory',
                         (('--memory_length', 200, int),
                          ('--sequence_length', 10, int),
                          ('--category_amount', 10, int),
                          ('--sample_amount', 100_000, int),
                          ('--loss_name', 'SparseCategoricalCrossentropy', str),
                          ('--loss_config', {'from_logits': True}, dict),
                          ('--metric_name', 'SparseCategoricalAccuracy', str)))

    def get_data_and_output_size(self):
        memory_length = self.args.memory_length
        sequence_length = self.args.sequence_length
        category_amount = self.args.category_amount
        sample_amount = self.args.sample_amount
        memory_sequence = np.random.randint(low=0, high=category_amount, size=(sample_amount, sequence_length, 1))
        first_blank_sequence = category_amount * np.ones((sample_amount, memory_length, 1))
        marker_sequence = np.random.randint(low=0, high=sequence_length, size=(sample_amount, 1, 1))
        input_sequence = np.concatenate((memory_sequence, first_blank_sequence, marker_sequence), 1)
        time_sequence = np.ones_like(input_sequence)
        output_sequence = memory_sequence[np.arange(sample_amount), np.squeeze(marker_sequence)]
        return (input_sequence, time_sequence), (output_sequence,), self.args.category_amount


MemoryBenchmark()
