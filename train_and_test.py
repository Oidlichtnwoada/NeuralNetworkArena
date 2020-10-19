from os import listdir
from os.path import join

from numpy import load, array
from numpy.random import random, shuffle
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import RMSprop

from models.continuous_transformer import ContinuousTransformer


class ProblemLoader:
    def __init__(self, problem_name='walker', sequence_length=64, skip_percentage=0.1, test_data_percentage=0.15, validation_data_percentage=0.1):
        self.problem_path = join('problems', problem_name)
        self.sequence_length = sequence_length
        self.skip_percentage = skip_percentage
        self.test_data_percentage = test_data_percentage
        self.validation_data_percentage = validation_data_percentage
        self.test_sequences = None
        self.validation_sequences = None
        self.training_sequences = None
        self.input_length = None

    def build_datasets(self):
        # return test, training and validation set
        data = self.load_datasets()
        lossy_data = self.skip_data(data)
        sequences = self.get_sequences(lossy_data)
        self.test_sequences, self.validation_sequences, self.training_sequences = self.split_data(sequences)

    def load_datasets(self):
        # load datasets from files
        datasets = []
        for dataset_filename in [x for x in listdir(self.problem_path) if x.endswith('.npy')]:
            dataset = load(join(self.problem_path, dataset_filename))
            if self.input_length is None:
                self.input_length = dataset.shape[1]
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
                    lossy_input_dataset.append(input_dataset[index - interval + 1])
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
        # partition data for training, validation and test
        sequences_length = len(sequences)
        test_data_length = int(sequences_length * self.test_data_percentage)
        validation_data_length = int(sequences_length * self.validation_data_percentage)
        test_sequences = sequences[:test_data_length]
        validation_sequences = sequences[test_data_length:test_data_length + validation_data_length]
        training_sequences = sequences[test_data_length + validation_data_length:]
        processed_sequences = []
        for sequence in [test_sequences, validation_sequences, training_sequences]:
            processed_sequences.append((array([x[0] for x in sequence]),
                                        array([x[1] for x in sequence]),
                                        array([x[2] for x in sequence])))
        return processed_sequences

    def train_and_test(self):
        # train the model parameters using gradient descent and test it afterwards
        model = ContinuousTransformer([(self.sequence_length, self.input_length), (self.sequence_length, 1)], [(self.sequence_length, self.input_length)])
        model.compile(optimizer=RMSprop(0.005), loss=MeanSquaredError())
        model.fit(
            x=(self.training_sequences[0], self.training_sequences[1]),
            y=self.training_sequences[2],
            batch_size=128,
            epochs=200,
            validation_data=((self.validation_sequences[0], self.validation_sequences[1]), self.validation_sequences[2]),
            callbacks=[ModelCheckpoint('weights/checkpoint', save_best_only=True, save_weights_only=True)],
        )
        model.load_weights('weights/checkpoint')
        best_test_loss = model.evaluate(x=(self.test_sequences[0], self.test_sequences[1]), y=self.test_sequences[2])
        print(f'best test loss: {best_test_loss}')


problem_loader = ProblemLoader()
problem_loader.build_datasets()
problem_loader.train_and_test()
