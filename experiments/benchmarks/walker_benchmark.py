import os

import numpy as np

import experiments.benchmarks.benchmark as benchmark


class WalkerBenchmark(benchmark.Benchmark):
    def __init__(self):
        super().__init__('walker', True, True, False,
                         (('--skip_percentage', 0.1, float),
                          ('--frame_skip', False, bool),
                          ('--sequence_length', 64, int),
                          ('--max_sample_amount', 10_000, int),
                          ('--loss_name', 'MeanSquaredError', str),
                          ('--loss_config', {}, dict),
                          ('--metric_name', 'MeanAbsoluteError', str)))

    def get_data(self):
        max_sample_amount = self.args.max_sample_amount
        datasets = []
        for dataset_filename in [x for x in os.listdir(self.supplementary_data_directory) if x.endswith('.npy')]:
            dataset = np.load(os.path.join(self.supplementary_data_directory, dataset_filename))
            datasets.append((dataset[:-1, :].tolist(), dataset[1:, :].tolist()))
        lossy_data = []
        for input_dataset, output_dataset in datasets:
            lossy_input_dataset, lossy_time_dataset, lossy_output_dataset = [], [], []
            interval = 0
            for index in range(len(input_dataset)):
                interval += 1
                if np.random.random() > self.args.skip_percentage or not self.args.frame_skip:
                    lossy_input_dataset.append(input_dataset[index])
                    lossy_time_dataset.append([interval])
                    lossy_output_dataset.append(output_dataset[index])
                    interval = 0
            lossy_data.append([lossy_input_dataset, lossy_time_dataset, lossy_output_dataset])
        sequences = []
        for input_dataset, time_dataset, output_dataset in lossy_data:
            for start_index in range(0, len(input_dataset) - self.args.sequence_length + 1, self.args.sequence_length // 4):
                end_index = start_index + self.args.sequence_length
                sequences.append([input_dataset[start_index: end_index],
                                  time_dataset[start_index: end_index],
                                  output_dataset[start_index: end_index]])
        reshaped_sequences = np.swapaxes(sequences, 0, 1)
        return reshaped_sequences[:2, :max_sample_amount], reshaped_sequences[2:, :max_sample_amount], len(reshaped_sequences[0][0][0])


WalkerBenchmark()
