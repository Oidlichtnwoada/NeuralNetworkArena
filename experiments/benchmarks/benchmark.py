import abc
import argparse
import collections.abc
import os
import shutil

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import tensorflow as tf

import experiments.models.model_factory as model_factory


class Benchmark(abc.ABC):
    def __init__(self, name, iterative_data, output_per_timestep, use_mask, parser_configs):
        self.name = name
        self.iterative_data = iterative_data
        self.models = model_factory.get_model_descriptions()
        self.output_per_timestep = output_per_timestep
        self.use_mask = use_mask
        self.args = self.get_args(parser_configs)
        self.project_directory = os.getcwd()
        self.saved_model_directory = os.path.join(self.project_directory, self.args.saved_model_folder_name, self.name)
        self.tensorboard_directory = os.path.join(self.project_directory, self.args.tensorboard_folder_name, self.name)
        self.supplementary_data_directory = os.path.join(self.project_directory, self.args.supplementary_data_folder_name, self.name)
        self.result_directory = os.path.join(self.project_directory, self.args.result_folder_name, self.name)
        self.visualization_directory = os.path.join(self.project_directory, self.args.visualization_folder_name, self.name)
        self.input_data, self.output_data, self.output_size = self.get_data()
        self.input_data, self.output_data = np.array(self.input_data), np.array(self.output_data)
        self.random_integer = np.random.randint(2 ** 30)
        np.random.default_rng(self.random_integer).shuffle(self.input_data, 1)
        np.random.default_rng(self.random_integer).shuffle(self.output_data, 1)
        self.preprocess_data()
        self.data_samples = self.input_data.shape[1]
        assert self.data_samples == self.output_data.shape[1]
        self.inputs = tuple((tf.keras.Input(shape=get_recursive_shape(x)[1:], batch_size=self.args.batch_size, dtype=self.get_correct_datatype(index)) for index, x in enumerate(self.input_data)))
        self.test_samples = int(self.data_samples * self.args.test_data_percentage)
        self.validation_samples = int(self.data_samples * self.args.validation_data_percentage)
        self.training_samples = self.data_samples - self.test_samples - self.validation_samples
        self.test_input_data = self.postprocess_data(self.input_data[:, :self.test_samples])
        self.validation_input_data = self.postprocess_data(self.input_data[:, self.test_samples:-self.training_samples])
        self.training_input_data = self.postprocess_data(self.input_data[:, -self.training_samples:])
        self.test_output_data = self.postprocess_data(self.output_data[:, :self.test_samples])
        self.validation_output_data = self.postprocess_data(self.output_data[:, self.test_samples:-self.training_samples])
        self.training_output_data = self.postprocess_data(self.output_data[:, -self.training_samples:])
        self.model, self.fit_result, self.evaluate_result = (None,) * 3
        self.train_and_test()

    def preprocess_data(self):
        if self.iterative_data and not self.models[self.args.model]:
            if self.output_per_timestep:
                sequence_length = self.input_data.shape[2]
                samples = self.input_data.shape[1] // self.args.shrink_divisor
                input_shapes = tuple((list(get_recursive_shape(x)) for x in self.input_data))
                output_shapes = tuple((list(get_recursive_shape(x[:, 0])) for x in self.output_data))
                for shape in input_shapes + output_shapes:
                    shape[0] = sequence_length * samples
                input_data_tuple = tuple((np.zeros(x) for x in input_shapes))
                output_data_tuple = tuple((np.zeros(x) for x in output_shapes))
                for sample_index in range(samples):
                    for subsequence_length in range(1, sequence_length + 1):
                        for input_index, input_data in enumerate(input_data_tuple):
                            input_data[sample_index * sequence_length + subsequence_length - 1, :subsequence_length] = \
                                get_numpy_array(self.input_data[input_index, sample_index, :subsequence_length])
                        for output_index, output_data in enumerate(output_data_tuple):
                            output_data[sample_index * sequence_length + subsequence_length - 1] = \
                                get_numpy_array(self.output_data[output_index, sample_index, subsequence_length - 1])
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
        return tuple((get_numpy_array(x) for x in data))

    def check_directories(self):
        shutil.rmtree(os.path.join(self.tensorboard_directory, self.args.model), ignore_errors=True)
        shutil.rmtree(os.path.join(self.result_directory, self.args.model), ignore_errors=True)
        for model_name in self.models:
            os.makedirs(os.path.join(self.saved_model_directory, model_name), exist_ok=True)
            os.makedirs(os.path.join(self.tensorboard_directory, model_name), exist_ok=True)
            os.makedirs(os.path.join(self.result_directory, model_name), exist_ok=True)
            os.makedirs(os.path.join(self.visualization_directory), exist_ok=True)

    def train_and_test(self):
        self.check_directories()
        model_name = self.args.model
        assert model_name in self.models
        model_save_location = os.path.join(self.saved_model_directory, model_name)
        model_tensorboard_location = os.path.join(self.tensorboard_directory, model_name)
        if self.args.use_saved_model:
            self.model = tf.keras.models.load_model(model_save_location)
        else:
            mask_tensor = self.inputs[-1] if self.use_mask else None
            input_tensor = self.inputs[:-1] if self.use_mask else self.inputs
            inputs_slice = slice(None) if self.args.use_time_input or len(input_tensor) == 1 else slice(-1)
            self.model = tf.keras.Model(inputs=self.inputs,
                                        outputs=model_factory.get_model_output_by_name(self.args.model, self.output_size,
                                                                                       input_tensor[inputs_slice], self.output_per_timestep, mask_tensor))
            optimizer = tf.keras.optimizers.get({'class_name': self.args.optimizer_name,
                                                 'config': {'learning_rate': self.args.learning_rate}})
            loss = tf.keras.losses.get({'class_name': self.args.loss_name,
                                        'config': self.args.loss_config})
            metric = tf.keras.metrics.get(self.args.metric_name)
            self.model.compile(optimizer=optimizer, loss=loss, metrics=metric, run_eagerly=self.args.debug)
        self.model.summary()
        if not self.args.skip_training:
            self.fit_result = self.model.fit(
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
        self.model = tf.keras.models.load_model(model_save_location)
        self.evaluate_result = self.model.evaluate(
            x=self.test_input_data,
            y=self.test_output_data,
            batch_size=self.args.batch_size,
            callbacks=(tf.keras.callbacks.TensorBoard(log_dir=model_tensorboard_location)),
            return_dict=True)
        if not self.args.skip_training:
            self.create_visualization()
            self.accumulate_data()

    def create_visualization(self):
        fit_results = list(self.fit_result.history.items())
        fit_header = self.correct_names([x[0] for x in fit_results], train=True)
        fit_data = np.array([x[1] for x in fit_results])
        fit_table = pd.DataFrame(data=fit_data.T, columns=fit_header)
        fit_table.to_csv(os.path.join(self.result_directory, self.args.model, 'training.csv'), index=False)
        evaluate_results = list(self.evaluate_result.items())
        evaluate_header = self.correct_names([x[0] for x in evaluate_results], train=False)
        evaluate_data = np.array([x[1] for x in evaluate_results])
        evaluate_table = pd.DataFrame(data=np.expand_dims(evaluate_data, 0), columns=evaluate_header)
        evaluate_table.insert(0, 'model', self.args.model)
        evaluate_table.insert(1, 'trainable parameters', np.sum([np.prod(x.shape) for x in self.model.trainable_variables]))
        evaluate_table.to_csv(os.path.join(self.result_directory, self.args.model, 'testing.csv'), index=False)
        fit_table.drop(fit_table.columns[-1], axis=1, inplace=True)
        x_data = np.array(range(1, max(self.fit_result.epoch) + 2)) * len(self.training_input_data[0])
        figure, first_axis = plt.subplots()
        first_axis.set_xlabel('training samples')
        first_axis.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        first_axis.set_title(f'{self.args.model.replace("_", " ")} @ {self.__class__.__name__}')
        second_axis = first_axis.twinx()
        axes = [first_axis, second_axis]
        for index, column in enumerate(fit_table.columns):
            axes[index % 2].plot(x_data, fit_table[column].tolist(), label=column)
        for index, column in enumerate(evaluate_table.columns[2:]):
            axes[index % 2].hlines(evaluate_table[column].tolist(), x_data[0], x_data[-1], label=column, linestyles='dashed', colors='black')
        first_axis.legend(loc='center left', prop={'size': 6})
        second_axis.legend(loc='center right', prop={'size': 6})
        plt.savefig(os.path.join(self.visualization_directory, f'{self.args.model}.pdf'))

    def accumulate_data(self):
        testing_data = []
        for model_name in self.models:
            test_results_path = os.path.join(self.result_directory, model_name, 'testing.csv')
            if os.path.exists(test_results_path):
                table = pd.read_csv(test_results_path)
                testing_data.append(table)
        merged_table = pd.concat(testing_data)
        merged_table.sort_values(merged_table.columns[2], inplace=True)
        merged_table.to_csv(os.path.join(self.result_directory, 'merged_results.csv'), index=False)

    def correct_names(self, names, train):
        loss_name = self.fit_result.model.loss.name
        lr_name = 'learning rate'
        corrected_names = []
        for name in names:
            if train:
                if not name.startswith('val'):
                    name = 'train ' + name
            else:
                name = 'test ' + name
            corrected_names.append(name.replace('loss', loss_name).replace('lr', lr_name).replace('_', ' '))
        return corrected_names

    def get_correct_datatype(self, index):
        if index == len(self.input_data) - 1 and self.use_mask:
            return tf.bool
        else:
            return tf.float32

    def get_args(self, parser_configs):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', default=list(self.models)[0], type=str)
        parser.add_argument('--epochs', default=1024, type=int)
        parser.add_argument('--batch_size', default=32, type=int)
        parser.add_argument('--optimizer_name', default='adam', type=str)
        parser.add_argument('--learning_rate', default=1E-3, type=float)
        parser.add_argument('--use_saved_model', default=False, type=bool)
        parser.add_argument('--debug', default=False, type=bool)
        parser.add_argument('--skip_training', default=False, type=bool)
        parser.add_argument('--validation_data_percentage', default=0.1, type=float)
        parser.add_argument('--test_data_percentage', default=0.1, type=float)
        parser.add_argument('--no_improvement_lr_patience', default=4, type=int)
        parser.add_argument('--no_improvement_abort_patience', default=10, type=int)
        parser.add_argument('--saved_model_folder_name', default='saved_models', type=str)
        parser.add_argument('--tensorboard_folder_name', default='tensorboard', type=str)
        parser.add_argument('--supplementary_data_folder_name', default='supplementary_data', type=str)
        parser.add_argument('--result_folder_name', default='results', type=str)
        parser.add_argument('--visualization_folder_name', default='visualizations', type=str)
        parser.add_argument('--shrink_divisor', default=1, type=int)
        parser.add_argument('--use_time_input', default=False, type=bool)
        for parser_config in parser_configs:
            argument_name, default, cls = parser_config
            parser.add_argument(argument_name, default=default, type=cls)
        return parser.parse_args()

    @abc.abstractmethod
    def get_data(self):
        raise NotImplementedError


def get_numpy_array(array):
    array = np.array(array)
    numpy_shape = array.shape
    actual_shape = get_recursive_shape(array)
    if numpy_shape != actual_shape:
        array = np.array(array.tolist())
    assert array.shape == actual_shape
    return array


def get_recursive_shape(array):
    if isinstance(array, collections.abc.Iterable) and isinstance(array, collections.abc.Sized):
        return (len(array),) + get_recursive_shape(array[0])
    else:
        return ()
