import numpy as np

import experiments.benchmarks.benchmark as benchmark


class AddBenchmark(benchmark.Benchmark):
    def __init__(self):
        super().__init__('add',
                         (('--sequence_length', 100, int),
                          ('--samples', 40_000, int),
                          ('--loss_name', 'MeanSquaredError', str),
                          ('--loss_config', {}, dict),
                          ('--metric_name', '', str)))

    def get_data_and_output_size(self):
        sequence_length = self.args.sequence_length
        assert sequence_length % 2 == 0
        samples = self.args.samples
        number_sequences = np.random.random((samples, sequence_length, 1))
        random_indices = np.random.randint(low=0, high=sequence_length // 2, size=2 * samples)
        row_indices = np.arange(samples)
        marker_sequences = np.zeros_like(number_sequences)
        marker_sequences[row_indices, random_indices[:samples]] = 1
        marker_sequences[row_indices, random_indices[samples:] + sequence_length // 2] = 1
        input_sequences = np.concatenate((number_sequences, marker_sequences), -1)
        time_sequences = np.ones((samples, sequence_length, 1))
        filtered_input_sequences = np.where(marker_sequences, number_sequences, 0)
        output_data = np.sum(filtered_input_sequences, 1)
        return (input_sequences, time_sequences), (output_data,), 1


AddBenchmark()
