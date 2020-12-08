from argparse import ArgumentParser
from os.path import join

from numpy import concatenate, stack, argmax, squeeze
from numpy import ones, ones_like, sum, mean
from numpy.random import randint
from tensorflow import math
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import RNN, Dense, LSTM, TimeDistributed
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, RMSprop

from models.memory_layer import MemoryLayerCell
from models.memory_rnn import NTMCell
from models.unitary_rnn import EUNNCell


class MemoryProblemLoader:
    def __init__(self, model, use_saved_weights, memory_length, sequence_length, category_amount, sample_amount, batch_size, epochs, learning_rate, debug,
                 test_data_percentage=0.15, validation_data_percentage=0.1):
        self.model = model
        self.use_saved_weights = use_saved_weights
        self.memory_length = memory_length
        self.sequence_length = sequence_length
        self.category_amount = category_amount
        self.sample_amount = sample_amount
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.debug = debug
        self.test_data_percentage = test_data_percentage
        self.validation_data_percentage = validation_data_percentage
        self.sample_length = self.memory_length + 2 * self.sequence_length
        self.loss_object = SparseCategoricalCrossentropy(from_logits=True)
        self.sample_weight = ones((1, self.sample_length,))
        self.sample_weight[:, self.sample_length - self.sequence_length] /= self.sample_length - self.sequence_length
        self.test_sequences = None
        self.validation_sequences = None
        self.training_sequences = None
        self.weights_directory = join('weights', 'memory', f'{self.model}', 'checkpoint')

    def get_memory_data(self):
        # build the data for the memory problem
        memory_sequence = randint(low=0, high=self.category_amount - 2, size=(self.sample_amount, self.sequence_length, 1))
        first_blank_sequence = (self.category_amount - 2) * ones((self.sample_amount, self.memory_length - 1, 1))
        marker_sequence = (self.category_amount - 1) * ones((self.sample_amount, 1, 1))
        second_blank_sequence = (self.category_amount - 2) * ones((self.sample_amount, self.sequence_length, 1))
        input_sequence = concatenate((memory_sequence, first_blank_sequence, marker_sequence, second_blank_sequence), 1)
        time_sequence = ones_like(input_sequence)
        third_blank_sequence = (self.category_amount - 2) * ones((self.sample_amount, self.memory_length + self.sequence_length, 1))
        output_sequence = concatenate((third_blank_sequence, memory_sequence), 1)
        return stack((input_sequence, time_sequence, output_sequence))

    def build_datasets(self):
        # return test, training and validation set
        self.test_sequences, self.validation_sequences, self.training_sequences = self.split_data(self.get_memory_data())

    def split_data(self, data):
        # partition data for training, validation and test
        test_sample_amount = int(self.sample_amount * self.test_data_percentage)
        validation_sample_amount = int(self.sample_amount * self.validation_data_percentage)
        test_samples = data[:, :test_sample_amount, :]
        validation_samples = data[:, test_sample_amount:test_sample_amount + validation_sample_amount, :]
        training_samples = data[:, test_sample_amount + validation_sample_amount:, :]
        return test_samples, validation_samples, training_samples

    def get_model(self):
        # build the model for the memory task
        inputs = (Input(shape=(self.sample_length, 1)), Input(shape=(self.sample_length, 1)))
        if self.model == 'memory_layer':
            outputs = RNN(MemoryLayerCell(100, self.category_amount), return_sequences=True)(inputs)
            optimizer = Adam(self.learning_rate)
        elif self.model == 'lstm':
            outputs = TimeDistributed(Dense(self.category_amount))(LSTM(40, return_sequences=True)(inputs[0]))
            optimizer = RMSprop(self.learning_rate)
        elif self.model == 'recurrent_memory_cell':
            outputs = RNN(NTMCell(1, 100, 128, 20, 1, 1, output_dim=self.category_amount), return_sequences=True)(inputs[0])
            optimizer = Adam(self.learning_rate)
        elif self.model == 'unitary_rnn':
            outputs = TimeDistributed(Dense(self.category_amount))(math.real(RNN(EUNNCell(128, 4), return_sequences=True)(inputs[0])))
            optimizer = RMSprop(self.learning_rate)
        else:
            raise NotImplementedError()
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=optimizer, loss=self.custom_loss, run_eagerly=self.debug)
        print(f'sample predictions: {model.predict((self.test_sequences[0][:self.batch_size], self.test_sequences[1][:self.batch_size]), batch_size=self.batch_size)}')
        model.summary()
        return model

    def custom_loss(self, y_true, y_pred):
        return self.loss_object(y_true, y_pred, sample_weight=self.sample_weight)

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
        model.load_weights(self.weights_directory)
        test_loss = model.evaluate(x=(self.test_sequences[0], self.test_sequences[1]), y=self.test_sequences[2], batch_size=self.batch_size)
        print(f'test loss: {test_loss:.4f}')
        # compute percentage of correct labels if argmax of output is taken
        predictions = argmax(model.predict((self.test_sequences[0], self.test_sequences[1]), batch_size=self.batch_size), -1)
        mean_correct_predictions = mean(sum((predictions == squeeze(self.test_sequences[2], -1)).astype(int), -1))
        print(f'mean correct predictions of memorized sequence: {mean_correct_predictions - self.memory_length - self.sequence_length:.1f}/{self.sequence_length:.1f}')


# parse arguments and start program
parser = ArgumentParser()
parser.add_argument('--mode', default='train', type=str)
parser.add_argument('--model', default='memory_layer', type=str)
parser.add_argument('--use_saved_weights', default=False, type=bool)
parser.add_argument('--memory_length', default=10, type=int)
parser.add_argument('--sequence_length', default=10, type=int)
parser.add_argument('--category_amount', default=10, type=int)
parser.add_argument('--sample_amount', default=100_000, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--epochs', default=256, type=int)
parser.add_argument('--learning_rate', default=1E-4, type=float)
parser.add_argument('--debug', default=False, type=bool)
args = parser.parse_args()

# build the problem loader using the arguments
problem_loader = MemoryProblemLoader(model=args.model, use_saved_weights=args.use_saved_weights, memory_length=args.memory_length, sequence_length=args.sequence_length,
                                     category_amount=args.category_amount, sample_amount=args.sample_amount, batch_size=args.batch_size,
                                     epochs=args.epochs, learning_rate=args.learning_rate, debug=args.debug)
problem_loader.build_datasets()
if args.mode == 'train':
    problem_loader.train()
elif args.mode == 'test':
    problem_loader.test()
