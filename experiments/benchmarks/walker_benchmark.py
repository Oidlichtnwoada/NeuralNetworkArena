import os

import numpy as np

import experiments.benchmarks.benchmark as benchmark


class WalkerBenchmark(benchmark.Benchmark):
    def __init__(self):
        super().__init__('walker',
                         (('--skip_percentage', 0.1, float),
                          ('--frame_skip', False, bool),
                          ('--sequence_length', 64, int),
                          ('--max_sample_amount', 50_000, int),
                          ('--sample_distance', 4, int),
                          ('--loss_name', 'MeanSquaredError', str),
                          ('--loss_config', {}, dict),
                          ('--metric_name', '', str)))

    def get_data_and_output_size(self):
        max_sample_amount = self.args.max_sample_amount
        datasets = []
        for dataset_filename in [x for x in os.listdir(self.supplementary_data_dir) if x.endswith('.npy')]:
            dataset = np.load(os.path.join(self.supplementary_data_dir, dataset_filename))
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
        input_sequences = []
        time_sequences = []
        output_sequences = []
        for input_dataset, time_dataset, output_dataset in lossy_data:
            for start_index in range(0, len(input_dataset) - self.args.sequence_length + 1, self.args.sample_distance):
                end_index = start_index + self.args.sequence_length
                input_sequences.append(input_dataset[start_index:end_index])
                time_sequences.append(time_dataset[start_index:end_index])
                output_sequences.append(output_dataset[end_index - 1])
        return (np.array(input_sequences[:max_sample_amount]), np.array(time_sequences[:max_sample_amount])), (np.array(output_sequences[:max_sample_amount]),), 17


WalkerBenchmark()
