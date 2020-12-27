import os

import numpy as np

from benchmark import Benchmark
from models.memory_layer import MemoryLayerAttention
from models.neural_circuit_policies import NeuralCircuitPolicies
from models.recurrent_transformer import MultiHeadRecurrentAttention
from models.transformer import Transformer, MultiHeadAttention


class WalkerBenchmark(Benchmark):
    def __init__(self):
        super().__init__('walker', True,
                         {'transformer': False,
                          'memory_layer_transformer': False,
                          'recurrent_transformer': False,
                          'neural_circuit_policies': True},
                         (('--skip_percentage', 0.1, float),
                          ('--sequence_length', 64, int),
                          ('--loss_name', 'MeanSquaredError', str),
                          ('--loss_config', {}, dict),
                          ('--metric_name', 'MeanSquaredError', str)))

    def get_data(self):
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
                if np.random.random() > self.args.skip_percentage:
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
        return reshaped_sequences[:2], reshaped_sequences[2:]

    def get_model_output(self, model):
        if model == 'transformer':
            return Transformer(token_amount=1, token_size=self.inputs[0].shape[2], d_model=64, num_heads=4, d_ff=128,
                               num_layers=4, dropout_rate=0.1, attention=MultiHeadAttention)(self.inputs)
        elif model == 'memory_layer_transformer':
            return Transformer(token_amount=1, token_size=self.inputs[0].shape[2], d_model=16, num_heads=2, d_ff=32,
                               num_layers=2, dropout_rate=0.1, attention=MemoryLayerAttention)(self.inputs)
        elif model == 'recurrent_transformer':
            return Transformer(token_amount=1, token_size=self.inputs[0].shape[2], d_model=32, num_heads=4, d_ff=64,
                               num_layers=1, dropout_rate=0.1, attention=MultiHeadRecurrentAttention)(self.inputs)
        elif model == 'neural_circuit_policies':
            return NeuralCircuitPolicies(
                output_length=self.inputs[0].shape[2], inter_neurons=32, command_neurons=16, motor_neurons=self.inputs[0].shape[2],
                sensory_fanout=4, inter_fanout=4, recurrent_command_synapses=8, motor_fanin=6)(self.inputs)


WalkerBenchmark()
