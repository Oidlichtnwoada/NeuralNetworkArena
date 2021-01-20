import numpy as np

import experiments.benchmarks.benchmark as benchmark


class AddBenchmark(benchmark.Benchmark):
    def __init__(self):
        super().__init__('memory',
                         (('--sequence_length', 200, int),
                          ('--sample_amount', 100_000, int),
                          ('--loss_name', 'MeanSquaredError', str),
                          ('--loss_config', {}, dict),
                          ('--metric_name', 'MeanAbsoluteError', str)))

    def get_data_and_output_size(self):
        sequence_length = self.args.sequence_length
        assert sequence_length % 2 == 0
        sample_amount = self.args.sample_amount
        number_sequences = np.random.random((sample_amount, sequence_length, 1))
        random_indices = np.random.randint(low=0, high=sequence_length // 2, size=2 * sample_amount)
        row_indices = np.arange(sample_amount)
        marker_sequences = np.zeros_like(number_sequences)
        marker_sequences[row_indices, random_indices[:sample_amount]] = 1
        marker_sequences[row_indices, random_indices[sample_amount:] + sequence_length // 2] = 1
        input_sequences = np.concatenate((number_sequences, marker_sequences), -1)
        filtered_input_sequences = np.where(marker_sequences, number_sequences, 0)
        output_data = np.sum(filtered_input_sequences, 1)
        return (input_sequences,), (output_data,), 1


AddBenchmark()
