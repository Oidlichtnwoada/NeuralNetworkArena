import os
from collections import defaultdict

import numpy as np
import pandas as pd

import experiments.benchmarks.benchmark as benchmark


class ActivityBenchmark(benchmark.Benchmark):
    def __init__(self):
        super().__init__('activity', True, True,
                         (('--sequence_length', 64, int),
                          ('--max_sample_amount', 3000, int),
                          ('--loss_name', 'SparseCategoricalCrossentropy', str),
                          ('--loss_config', {'from_logits': True}, dict),
                          ('--metric_name', 'SparseCategoricalAccuracy', str)))

    def get_data(self):
        sequence_length = self.args.sequence_length
        max_sample_amount = self.args.max_sample_amount
        activity_table = pd.read_csv(os.path.join(self.supplementary_data_directory, 'activity.csv'), header=None)
        activity_sets = defaultdict(list)
        last_sample_marker = None
        first_timestamp = 0
        last_time = 0
        for index, row in activity_table.iterrows():
            sample_marker = row[0]
            if sample_marker != last_sample_marker:
                last_sample_marker = sample_marker
                first_timestamp = row[2]
            row[2] -= first_timestamp
            row[2] /= 10 ** 7
            temp_last_time = last_time
            last_time = row[2]
            row[2] -= temp_last_time
            one_hot_encoding = [0, 0, 0, 0]
            one_hot_encoding[row[1]] = 1
            modified_row = [row[2]] + one_hot_encoding + row[3:].tolist()
            activity_sets[sample_marker].append(modified_row)
        sensor_inputs = np.zeros((0, sequence_length, 7))
        time_inputs = np.zeros((0, sequence_length, 1))
        activity_outputs = np.zeros((0, sequence_length, 1))
        for activity_set in activity_sets.values():
            activity_set_array = np.array(activity_set)
            for end_index in range(sequence_length + 1, len(activity_set_array), sequence_length):
                current_sequence = np.expand_dims(activity_set_array[end_index - sequence_length:end_index], 0)
                sensor_inputs = np.concatenate((sensor_inputs, current_sequence[..., 1:8]))
                time_inputs = np.concatenate((time_inputs, current_sequence[..., :1]))
                activity_outputs = np.concatenate((activity_outputs, current_sequence[..., 8:]))
        return (sensor_inputs[:max_sample_amount].tolist(), time_inputs[:max_sample_amount].tolist()), (activity_outputs[:max_sample_amount],), 7


ActivityBenchmark()
