from os import listdir
from os.path import join

from numpy import load
from numpy.random import random, shuffle


class ProblemLoader:
    def __init__(self, problem_name='walker', sequence_length=64, skip_percentage=0.1, test_data_percentage=0.15, validation_data_percentage=0.1):
        self.problem_path = join('problems', problem_name)
        self.sequence_length = sequence_length
        self.skip_percentage = skip_percentage
        self.test_data_percentage = test_data_percentage
        self.validation_data_percentage = validation_data_percentage

    def get_datasets(self):
        # return test, training and validation set
        data = self.load_datasets()
        lossy_data = self.skip_data(data)
        sequences = self.get_sequences(lossy_data)
        return self.split_data(sequences)

    def load_datasets(self):
        # load datasets from files
        datasets = []
        for dataset_filename in [x for x in listdir(self.problem_path) if x.endswith('.npy')]:
            dataset = load(join(self.problem_path, dataset_filename))
            datasets.append([dataset[:-1, :].tolist(), dataset[1:, :].tolist()])
        return datasets

    def skip_data(self, data):
        # skip 10% of all data
        lossy_data = []
        for input_dataset, output_dataset in data:
            lossy_input_dataset, lossy_time_dataset, lossy_output_dataset = [], [], []
            interval = 0
            for index in range(len(input_dataset)):
                interval += 1
                if random() > self.skip_percentage:
                    lossy_input_dataset.append(input_dataset[index])
                    lossy_time_dataset.append([interval])
                    lossy_output_dataset.append(output_dataset[index])
                    interval = 0
            lossy_data.append([lossy_input_dataset, lossy_time_dataset, lossy_output_dataset])
        return lossy_data

    def get_sequences(self, lossy_data):
        # generate training sequences of specific length
        sequences = []
        for input_dataset, time_dataset, output_dataset in lossy_data:
            for start_index in range(0, len(input_dataset) - self.sequence_length + 1, self.sequence_length // 4):
                end_index = start_index + self.sequence_length
                sequences.append([input_dataset[start_index: end_index],
                                  time_dataset[start_index: end_index],
                                  output_dataset[start_index: end_index]])
        shuffle(sequences)
        return sequences

    def split_data(self, sequences):
        sequences_length = len(sequences)
        test_data_length = int(sequences_length * self.test_data_percentage)
        validation_data_length = int(sequences_length * self.validation_data_percentage)
        test_sequences = sequences[:test_data_length]
        validation_sequences = sequences[test_data_length:test_data_length + validation_data_length]
        training_sequences = sequences[test_data_length + validation_data_length:]
        return test_sequences, validation_sequences, training_sequences


test_seqs, validation_seqs, training_seqs = ProblemLoader().get_datasets()
