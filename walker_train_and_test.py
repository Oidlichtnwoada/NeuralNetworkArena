from argparse import ArgumentParser
from os import listdir
from os.path import join

from numpy import load, array, zeros
from numpy.random import random, shuffle
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam, RMSprop

from models.memory_layer import MemoryLayerAttention
from models.neural_circuit_policies import NeuralCircuitPolicies
from models.recurrent_transformer import MultiHeadRecurrentAttention
from models.transformer import Transformer, MultiHeadAttention


class WalkerProblemLoader:
    def __init__(self, model, use_saved_weights, shrink_divisor, sequence_length, batch_size, epochs, learning_rate, debug,
                 skip_percentage=0.1, test_data_percentage=0.15, validation_data_percentage=0.1):
        self.problem_path = join('problems', 'walker')
        self.use_saved_weights = use_saved_weights
        self.shrink_divisor = shrink_divisor
        self.sequence_length = sequence_length
        self.skip_percentage = skip_percentage
        self.test_data_percentage = test_data_percentage
        self.validation_data_percentage = validation_data_percentage
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.debug = debug
        self.test_sequences = None
        self.validation_sequences = None
        self.training_sequences = None
        self.input_length = None
        self.model = model
        self.weights_directory = join('weights', 'walker', f'{self.model}', 'checkpoint')

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
        # partition data for training, validation and test
        sequences_length = len(sequences)
        test_data_length = int(sequences_length * self.test_data_percentage)
        validation_data_length = int(sequences_length * self.validation_data_percentage)
        test_sequences = sequences[:test_data_length]
        validation_sequences = sequences[test_data_length:test_data_length + validation_data_length]
        training_sequences = sequences[test_data_length + validation_data_length:]
        processed_sequences = []
        for sequence in [test_sequences, validation_sequences, training_sequences]:
            processed_sequences.append([array([x[0] for x in sequence]),
                                        array([x[1] for x in sequence]),
                                        array([x[2] for x in sequence])])
        return processed_sequences

    def transform_sequences(self):
        # transform rnn training sequences to transformer training sequences
        test_sequences, validation_sequences, training_sequences = [[], [], []], [[], [], []], [[], [], []]
        for input_sequences, output_sequences in [(self.test_sequences, test_sequences), (self.validation_sequences, validation_sequences), (self.training_sequences, training_sequences)]:
            # create a transformer training sample for each memory length and sequence (shrink to fit in RAM)
            for sequence_index in range(len(input_sequences[0]) // self.shrink_divisor):
                for memory_length in range(1, self.sequence_length + 1):
                    # zero pad information from future events
                    input_sequence = input_sequences[0][sequence_index].copy()
                    input_sequence[memory_length:] = zeros(input_sequence[memory_length:].shape)
                    output_sequences[0].append(input_sequence)
                    # zero pad time intervals from future events
                    time_intervals = input_sequences[1][sequence_index].copy()
                    time_intervals[memory_length:] = zeros(time_intervals[memory_length:].shape)
                    output_sequences[1].append(time_intervals)
                    # only use the next state of the last non-zero state in the input data as expected output data
                    output_sequences[2].append(input_sequences[2][sequence_index][memory_length - 1].copy())
            # convert sequences to numpy arrays
            output_sequences[0], output_sequences[1], output_sequences[2] = array(output_sequences[0]), array(output_sequences[1]), array(output_sequences[2])
        # update the instance properties
        self.test_sequences, self.validation_sequences, self.training_sequences = test_sequences, validation_sequences, training_sequences

    def get_model(self):
        loss = MeanSquaredError()
        if self.model == 'transformer':
            self.transform_sequences()
            model = Transformer(token_amount=1, token_size=self.input_length, d_model=64, num_heads=4, d_ff=128, num_layers=4, dropout_rate=0.1, attention=MultiHeadAttention)
            optimizer = Adam(self.learning_rate)
        elif self.model == 'neural_circuit_policies':
            model = NeuralCircuitPolicies(
                output_length=self.input_length, inter_neurons=16, command_neurons=16, motor_neurons=self.input_length,
                sensory_fanout=4, inter_fanout=4, recurrent_command_synapses=8, motor_fanin=6)
            optimizer = RMSprop(self.learning_rate)
        elif self.model == 'recurrent_transformer':
            self.transform_sequences()
            model = Transformer(token_amount=1, token_size=self.input_length, d_model=32, num_heads=4, d_ff=64, num_layers=1, dropout_rate=0.1, attention=MultiHeadRecurrentAttention)
            optimizer = RMSprop(self.learning_rate)
        elif self.model == 'memory_layer_transformer':
            self.transform_sequences()
            model = Transformer(token_amount=1, token_size=self.input_length, d_model=16, num_heads=2, d_ff=32,
                                num_layers=2, dropout_rate=0.1, attention=MemoryLayerAttention)
            optimizer = Adam(self.learning_rate)
        else:
            raise NotImplementedError()
        model.compile(optimizer=optimizer, loss=loss, run_eagerly=self.debug)
        print(f'sample predictions: {model.predict((self.test_sequences[0][:self.batch_size], self.test_sequences[1][:self.batch_size]), batch_size=self.batch_size)}')
        model.summary()
        return model

    def train(self):
        # train the model parameters using gradient descent
        model = self.get_model()
        if self.use_saved_weights:
            model.load_weights(self.weights_directory).expect_partial()
        model.fit(
            x=(self.training_sequences[0], self.training_sequences[1]),
            y=self.training_sequences[2],
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=((self.validation_sequences[0], self.validation_sequences[1]), self.validation_sequences[2]),
            callbacks=[ModelCheckpoint(self.weights_directory, save_best_only=True, save_weights_only=True)]
        )

    def test(self):
        # evaluate the loss on the test dataset
        model = self.get_model()
        model.load_weights(self.weights_directory).expect_partial()
        test_loss = model.evaluate(x=(self.test_sequences[0], self.test_sequences[1]), y=self.test_sequences[2], batch_size=self.batch_size)
        print(f'test loss: {test_loss}')


# parse arguments and start program
parser = ArgumentParser()
parser.add_argument('--model', default='transformer', type=str)
parser.add_argument('--mode', default='train', type=str)
parser.add_argument('--use_saved_weights', default=False, type=bool)
parser.add_argument('--debug', default=False, type=bool)
parser.add_argument('--shrink_divisor', default=1, type=int)
parser.add_argument('--sequence_length', default=64, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--epochs', default=256, type=int)
parser.add_argument('--learning_rate', default=1E-3, type=float)
args = parser.parse_args()

# build the problem loader using the arguments
problem_loader = WalkerProblemLoader(model=args.model, use_saved_weights=args.use_saved_weights, shrink_divisor=args.shrink_divisor,
                                     sequence_length=args.sequence_length, batch_size=args.batch_size, epochs=args.epochs, learning_rate=args.learning_rate, debug=args.debug)
problem_loader.build_datasets()
if args.mode == 'train':
    problem_loader.train()
elif args.mode == 'test':
    problem_loader.test()
