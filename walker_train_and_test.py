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
                         (('--model', 'transformer', str),
                          ('--shrink_divisor', 1, int),
                          ('--skip_percentage', 0.1, float),
                          ('--sequence_length', 64, int),
                          ('--loss_name', 'MeanSquaredError', str),
                          ('--loss_config', {}, dict),
                          ('--metric_name', 'MeanSquaredError', str)))
        self.add_model_output('transformer', Transformer(token_amount=1, token_size=self.inputs[0].shape[2], d_model=64, num_heads=4, d_ff=128,
                                                         num_layers=4, dropout_rate=0.1, attention=MultiHeadAttention)(self.inputs), False)
        self.add_model_output('memory_layer_transformer', Transformer(token_amount=1, token_size=self.inputs[0].shape[2], d_model=16, num_heads=2, d_ff=32,
                                                                      num_layers=2, dropout_rate=0.1, attention=MemoryLayerAttention)(self.inputs), False)
        self.add_model_output('recurrent_transformer', Transformer(token_amount=1, token_size=self.inputs[0].shape[2], d_model=32, num_heads=4, d_ff=64,
                                                                   num_layers=1, dropout_rate=0.1, attention=MultiHeadRecurrentAttention)(self.inputs), False)
        self.add_model_output('neural_circuit_policies', NeuralCircuitPolicies(
            output_length=self.inputs[0].shape[2], inter_neurons=32, command_neurons=16, motor_neurons=self.inputs[0].shape[2],
            sensory_fanout=4, inter_fanout=4, recurrent_command_synapses=8, motor_fanin=6)(self.inputs), True)
        self.train_and_test()

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

    def transform_sequences(self):
        # transform rnn training sequences to transformer training sequences
        test_sequences, validation_sequences, training_sequences = [[], [], []], [[], [], []], [[], [], []]
        for input_sequences, output_sequences in [(self.test_sequences, test_sequences), (self.validation_sequences, validation_sequences), (self.training_sequences, training_sequences)]:
            # create a transformer training sample for each memory length and sequence (shrink to fit in RAM)
            for sequence_index in range(len(input_sequences[0]) // self.shrink_divisor):
                for memory_length in range(1, self.sequence_length + 1):
                    # zero pad information from future events
                    input_sequence = input_sequences[0][sequence_index].copy()
                    input_sequence[memory_length:] = np.zeros(input_sequence[memory_length:].shape)
                    output_sequences[0].append(input_sequence)
                    # zero pad time intervals from future events
                    time_intervals = input_sequences[1][sequence_index].copy()
                    time_intervals[memory_length:] = np.zeros(time_intervals[memory_length:].shape)
                    output_sequences[1].append(time_intervals)
                    # only use the next state of the last non-zero state in the input data as expected output data
                    output_sequences[2].append(input_sequences[2][sequence_index][memory_length - 1].copy())
            # convert sequences to numpy arrays
            output_sequences[0], output_sequences[1], output_sequences[2] = np.array(output_sequences[0]), np.array(output_sequences[1]), np.array(output_sequences[2])
        # update the instance properties
        self.test_sequences, self.validation_sequences, self.training_sequences = test_sequences, validation_sequences, training_sequences


WalkerBenchmark()
