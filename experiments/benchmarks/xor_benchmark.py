import numpy as np

import experiments.benchmarks.benchmark as benchmark


class XorBenchmark(benchmark.Benchmark):
    def __init__(self):
        super().__init__('memory', True, False, True,
                         (('--sequence_length', 32, int),
                          ('--sample_amount', 100_000, int),
                          ('--loss_name', 'SparseCategoricalCrossentropy', str),
                          ('--loss_config', {'from_logits': True}, dict),
                          ('--metric_name', 'SparseCategoricalAccuracy', str)))

    def get_data(self):
        sequence_length = self.args.sequence_length
        sample_amount = self.args.sample_amount
        shape = (sample_amount, sequence_length, 1)
        first_part_masks = np.ones((sample_amount, 2))
        second_part_masks = np.repeat(np.tril(np.ones((sequence_length - 1, sequence_length - 2)), -1), sample_amount // (sequence_length - 1) + 1, 0)[:sample_amount]
        masks = np.concatenate((first_part_masks, second_part_masks), 1)
        masks = masks[..., np.newaxis].astype(bool)
        input_data = np.random.randint(2, size=shape)
        input_data = np.where(masks, input_data, 0)
        time_data = np.ones_like(input_data)
        time_data = np.where(masks, time_data, 0)
        output_data = np.sum(input_data, 1) % 2
        return np.stack((input_data, time_data, masks)), np.stack((output_data,)), 2


XorBenchmark()
