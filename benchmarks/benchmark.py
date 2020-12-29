import os
import shutil
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from collections.abc import Sized, Iterable

import numpy as np
import tensorflow as tf

from models.model_factory import get_model_output_by_name, get_model_descriptions


class Benchmark(ABC):
    def __init__(self, name, iterative_data, parser_configs):
        self.name = name
        self.iterative_data = iterative_data
        self.models = get_model_descriptions()
        self.args = self.get_args(parser_configs)
        self.project_directory = os.getcwd()
        self.saved_model_directory = os.path.join(self.project_directory, self.args.saved_model_folder_name, self.name)
        self.tensorboard_directory = os.path.join(self.project_directory, self.args.tensorboard_folder_name, self.name)
        self.supplementary_data_directory = os.path.join(self.project_directory, self.args.supplementary_data_folder_name, self.name)
        self.input_data, self.output_data, self.model_output_params = self.get_data()
        self.input_data, self.output_data = np.array(self.input_data), np.array(self.output_data)
        self.random_integer = np.random.randint(2 ** 30)
        np.random.default_rng(self.random_integer).shuffle(self.input_data, 1)
        np.random.default_rng(self.random_integer).shuffle(self.output_data, 1)
        self.preprocess_data()
        self.data_samples = self.input_data.shape[1]
        assert self.data_samples == self.output_data.shape[1]
        self.inputs = tuple((tf.keras.Input(shape=self.get_recursive_shape(x)[1:], batch_size=self.args.batch_size) for x in self.input_data))
        self.test_samples = int(self.data_samples * self.args.test_data_percentage)
        self.validation_samples = int(self.data_samples * self.args.validation_data_percentage)
        self.training_samples = self.data_samples - self.test_samples - self.validation_samples
        self.test_input_data = self.postprocess_data(self.input_data[:, :self.test_samples])
        self.validation_input_data = self.postprocess_data(self.input_data[:, self.test_samples:-self.training_samples])
        self.training_input_data = self.postprocess_data(self.input_data[:, -self.training_samples:])
        self.test_output_data = self.postprocess_data(self.output_data[:, :self.test_samples])
        self.validation_output_data = self.postprocess_data(self.output_data[:, self.test_samples:-self.training_samples])
        self.training_output_data = self.postprocess_data(self.output_data[:, -self.training_samples:])
        self.train_and_test()

    def preprocess_data(self):
        if self.iterative_data and not self.models[self.args.model]:
            sequence_length = self.input_data.shape[2]
            samples = self.input_data.shape[1] // self.args.shrink_divisor
            input_shapes = tuple((list(self.get_recursive_shape(x)) for x in self.input_data))
            output_shapes = tuple((list(self.get_recursive_shape(x[:, 0])) for x in self.output_data))
            for shape in input_shapes + output_shapes:
                shape[0] = sequence_length * samples
            input_data_tuple = tuple((np.zeros(x) for x in input_shapes))
            output_data_tuple = tuple((np.zeros(x) for x in output_shapes))
            for sample_index in range(samples):
                for subsequence_length in range(1, sequence_length + 1):
                    for input_index, input_data in enumerate(input_data_tuple):
                        input_data[sample_index * sequence_length + subsequence_length - 1, :subsequence_length] = \
                            self.get_numpy_array(self.input_data[input_index, sample_index, :subsequence_length])
                    for output_index, output_data in enumerate(output_data_tuple):
                        output_data[sample_index * sequence_length + subsequence_length - 1] = \
                            self.get_numpy_array(self.output_data[output_index, sample_index, subsequence_length - 1])
            del self.input_data, self.output_data
            input_data_list = []
            for input_data in input_data_tuple:
                input_data_list.append(input_data.tolist())
            self.input_data = np.array(input_data_list)
            del input_data_tuple, input_data_list
            output_data_list = []
            for output_data in output_data_tuple:
                output_data_list.append(output_data.tolist())
            self.output_data = np.array(output_data_list)
            del output_data_tuple, output_data_list
        elif not self.iterative_data and self.models[self.args.model]:
            raise NotImplementedError
        else:
            assert self.iterative_data == self.models[self.args.model]

    def postprocess_data(self, data):
        data_samples = data.shape[1]
        assert data_samples >= self.args.batch_size
        elements_to_remove = data_samples % self.args.batch_size
        if elements_to_remove != 0:
            data = data[:, :-elements_to_remove]
        return tuple((self.get_numpy_array(x) for x in data))

    def check_directories(self):
        shutil.rmtree(os.path.join(self.tensorboard_directory, self.args.model), ignore_errors=True)
        for model_name in self.models:
            os.makedirs(os.path.join(self.saved_model_directory, model_name), exist_ok=True)
            os.makedirs(os.path.join(self.tensorboard_directory, model_name), exist_ok=True)

    def train_and_test(self):
        self.check_directories()
        model_name = self.args.model
        assert model_name in self.models
        model_save_location = os.path.join(self.saved_model_directory, model_name)
        model_tensorboard_location = os.path.join(self.tensorboard_directory, model_name)
        if self.args.use_saved_model:
            model = tf.keras.models.load_model(model_save_location)
        else:
            model = tf.keras.Model(inputs=self.inputs,
                                   outputs=get_model_output_by_name(self.args.model, self.model_output_params[0], self.inputs[self.model_output_params[1]]))
            optimizer = tf.keras.optimizers.get({'class_name': self.args.optimizer_name,
                                                 'config': {'learning_rate': self.args.learning_rate}})
            loss = tf.keras.losses.get({'class_name': self.args.loss_name,
                                        'config': self.args.loss_config})
            metric = tf.keras.metrics.get(self.args.metric_name)
            model.compile(optimizer=optimizer, loss=loss, metrics=metric, run_eagerly=self.args.debug)
        model.summary()
        if not self.args.skip_training:
            model.fit(
                x=self.training_input_data,
                y=self.training_output_data,
                batch_size=self.args.batch_size,
                epochs=self.args.epochs,
                validation_data=(self.validation_input_data, self.validation_output_data),
                callbacks=(tf.keras.callbacks.ModelCheckpoint(model_save_location, save_best_only=True),
                           tf.keras.callbacks.EarlyStopping(patience=self.args.no_improvement_abort_patience),
                           tf.keras.callbacks.TerminateOnNaN(),
                           tf.keras.callbacks.ReduceLROnPlateau(patience=self.args.no_improvement_lr_patience),
                           tf.keras.callbacks.TensorBoard(log_dir=model_tensorboard_location)))
        model.evaluate(
            x=self.test_input_data,
            y=self.test_output_data,
            batch_size=self.args.batch_size,
            callbacks=(tf.keras.callbacks.TensorBoard(log_dir=model_tensorboard_location)))

    @staticmethod
    def get_numpy_array(array):
        array = np.array(array)
        numpy_shape = array.shape
        actual_shape = Benchmark.get_recursive_shape(array)
        if numpy_shape != actual_shape:
            array = np.array(array.tolist())
        assert array.shape == actual_shape
        return array

    @staticmethod
    def get_recursive_shape(array):
        if isinstance(array, Iterable) and isinstance(array, Sized):
            return (len(array),) + Benchmark.get_recursive_shape(array[0])
        else:
            return ()

    def get_args(self, parser_configs):
        parser = ArgumentParser()
        parser.add_argument('--model', default=list(self.models)[0], type=str)
        parser.add_argument('--epochs', default=256, type=int)
        parser.add_argument('--batch_size', default=32, type=int)
        parser.add_argument('--optimizer_name', default='adam', type=str)
        parser.add_argument('--learning_rate', default=0.001, type=float)
        parser.add_argument('--use_saved_model', default=False, type=bool)
        parser.add_argument('--debug', default=False, type=bool)
        parser.add_argument('--skip_training', default=False, type=bool)
        parser.add_argument('--validation_data_percentage', default=0.1, type=float)
        parser.add_argument('--test_data_percentage', default=0.1, type=float)
        parser.add_argument('--no_improvement_lr_patience', default=2, type=int)
        parser.add_argument('--no_improvement_abort_patience', default=4, type=int)
        parser.add_argument('--saved_model_folder_name', default='saved_models', type=str)
        parser.add_argument('--tensorboard_folder_name', default='tensorboard', type=str)
        parser.add_argument('--supplementary_data_folder_name', default='supplementary_data', type=str)
        parser.add_argument('--shrink_divisor', default=8, type=int)
        for parser_config in parser_configs:
            argument_name, default, cls = parser_config
            parser.add_argument(argument_name, default=default, type=cls)
        return parser.parse_args()

    @abstractmethod
    def get_data(self):
        raise NotImplementedError
