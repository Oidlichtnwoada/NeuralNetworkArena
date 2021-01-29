import abc
import argparse
import os
import shutil
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import tensorflow as tf

import experiments.models.model_factory as model_factory


class Benchmark(abc.ABC):
    def __init__(self, name, parser_configs):
        self.name = name
        self.args = self.get_args(parser_configs)
        self.saved_model_dir, self.tensorboard_dir, self.supplementary_data_dir, self.result_dir, self.visualization_dir = self.create_directories()
        self.input_data, self.output_data, self.output_size = self.get_data_and_output_size()
        self.data_samples = self.shuffle_data_and_return_sample_amount()
        self.inputs = tuple((tf.keras.Input(shape=x.shape[1:], batch_size=self.args.batch_size) for x in self.input_data))
        self.test_samples, self.validation_samples, self.training_samples = self.compute_sample_partition()
        self.test_input_data, self.validation_input_data, self.training_input_data = self.process_data(self.input_data)
        self.test_output_data, self.validation_output_data, self.training_output_data = self.process_data(self.output_data)
        self.train_and_test()

    @staticmethod
    def get_args(parser_configs):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', default=model_factory.MODEL_ARGUMENTS[0], type=str)
        parser.add_argument('--epochs', default=128, type=int)
        parser.add_argument('--batch_size', default=128, type=int)
        parser.add_argument('--optimizer_name', default='adam', type=str)
        parser.add_argument('--learning_rate', default=1E-3, type=float)
        parser.add_argument('--use_saved_model', default=False, type=bool)
        parser.add_argument('--debug', default=False, type=bool)
        parser.add_argument('--validation_data_percentage', default=0.1, type=float)
        parser.add_argument('--test_data_percentage', default=0.1, type=float)
        parser.add_argument('--no_improvement_lr_patience', default=4, type=int)
        parser.add_argument('--no_improvement_abort_patience', default=10, type=int)
        parser.add_argument('--saved_model_folder_name', default='saved_models', type=str)
        parser.add_argument('--tensorboard_folder_name', default='tensorboard', type=str)
        parser.add_argument('--supplementary_data_folder_name', default='supplementary_data', type=str)
        parser.add_argument('--result_folder_name', default='results', type=str)
        parser.add_argument('--visualization_folder_name', default='visualizations', type=str)
        parser.add_argument('--use_time_input', default=False, type=bool)
        for parser_config in parser_configs:
            argument_name, default, cls = parser_config
            parser.add_argument(argument_name, default=default, type=cls)
        return parser.parse_args()

    def create_directories(self):
        project_directory = os.getcwd()
        saved_model_directory = os.path.join(project_directory, self.args.saved_model_folder_name, self.name)
        tensorboard_directory = os.path.join(project_directory, self.args.tensorboard_folder_name, self.name)
        supplementary_data_directory = os.path.join(project_directory, self.args.supplementary_data_folder_name, self.name)
        result_directory = os.path.join(project_directory, self.args.result_folder_name, self.name)
        visualization_directory = os.path.join(project_directory, self.args.visualization_folder_name, self.name)
        return saved_model_directory, tensorboard_directory, supplementary_data_directory, result_directory, visualization_directory

    @abc.abstractmethod
    def get_data_and_output_size(self):
        raise NotImplementedError

    def shuffle_data_and_return_sample_amount(self):
        data_samples = None
        random_integer = np.random.randint(2 ** 30)
        for dataset in self.input_data + self.output_data:
            np.random.default_rng(random_integer).shuffle(dataset)
            data_samples = len(dataset)
        for dataset in self.input_data + self.output_data:
            assert data_samples == len(dataset)
        return data_samples

    def compute_sample_partition(self):
        test_samples = int(self.data_samples * self.args.test_data_percentage)
        test_samples -= test_samples % self.args.batch_size
        validation_samples = int(self.data_samples * self.args.validation_data_percentage)
        validation_samples -= validation_samples % self.args.batch_size
        training_samples = self.data_samples - test_samples - validation_samples
        training_samples -= training_samples % self.args.batch_size
        assert test_samples > 0 and validation_samples > 0 and training_samples > 0
        return test_samples, validation_samples, training_samples

    def process_data(self, data):
        test_data = tuple((x[slice(None, self.test_samples)] for x in data))
        validation_data = tuple((x[slice(self.test_samples, self.test_samples + self.validation_samples)] for x in data))
        training_data = tuple((x[slice(self.test_samples + self.validation_samples, self.test_samples + self.validation_samples + self.training_samples)] for x in data))
        return test_data, validation_data, training_data

    def check_directories(self):
        shutil.rmtree(os.path.join(self.tensorboard_dir, self.args.model), ignore_errors=True)
        for model_name in model_factory.MODEL_ARGUMENTS:
            os.makedirs(os.path.join(self.saved_model_dir, model_name), exist_ok=True)
            os.makedirs(os.path.join(self.tensorboard_dir, model_name), exist_ok=True)
            os.makedirs(os.path.join(self.result_dir, model_name), exist_ok=True)
            os.makedirs(os.path.join(self.visualization_dir), exist_ok=True)

    @staticmethod
    def correct_names(names, train, model):
        loss_name = model.loss.name
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

    def create_and_save_tables(self, model, fit_result, evaluate_result, training_duration):
        fit_results = list(fit_result.history.items())
        fit_header = self.correct_names([x[0] for x in fit_results], train=True, model=model)
        fit_data = np.array([x[1] for x in fit_results])
        fit_table = pd.DataFrame(data=fit_data.T, columns=fit_header)
        fit_table.to_csv(os.path.join(self.result_dir, self.args.model, 'training.csv'), index=False)
        fit_table.drop(fit_table.columns[-1], axis=1, inplace=True)
        evaluate_results = list(evaluate_result.items())
        evaluate_header = self.correct_names([x[0] for x in evaluate_results], train=False, model=model)
        evaluate_data = np.array([x[1] for x in evaluate_results])
        evaluate_table = pd.DataFrame(data=np.expand_dims(evaluate_data, 0), columns=evaluate_header)
        evaluate_table.insert(0, 'model', self.args.model)
        evaluate_table.insert(1, 'trainable parameters', np.sum([np.prod(x.shape) for x in model.trainable_variables]))
        evaluate_table.insert(2, 'training duration total', training_duration)
        evaluate_table.insert(3, 'training duration per epoch', training_duration / fit_table.shape[0])
        evaluate_table.insert(4, 'epochs', fit_table.shape[0])
        evaluate_table.to_csv(os.path.join(self.result_dir, self.args.model, 'testing.csv'), index=False)
        evaluate_table.drop(evaluate_table.columns[:5], axis=1, inplace=True)
        return fit_table, evaluate_table

    def create_visualization(self, fit_table, evaluate_table):
        x_data = np.array(range(1, fit_table.shape[0] + 1))
        figure, first_axis = plt.subplots()
        first_axis.set_xlabel('epochs')
        first_axis.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        first_axis.set_title(f'{self.args.model.replace("_", " ")} @ {self.__class__.__name__}')
        first_axis.set_prop_cycle(color=['red', 'green'])
        if self.args.metric_name == '':
            axes = [first_axis]
        else:
            second_axis = first_axis.twinx()
            second_axis.set_prop_cycle(color=['blue', 'orange'])
            second_axis.legend(loc='center right', prop={'size': 6})
            axes = [first_axis, second_axis]
        hline_colors = ['black', 'grey']
        legend_positions = ['center left', 'center right']
        for index, column in enumerate(fit_table.columns):
            axes[index % len(axes)].plot(x_data, fit_table[column].tolist(), label=column)
        for index, column in enumerate(evaluate_table.columns):
            axes[index % len(axes)].hlines(evaluate_table[column].tolist(), x_data[0], x_data[-1], label=column, linestyles='dashed', colors=hline_colors[index % len(hline_colors)])
        for index, axis in enumerate(axes):
            axis.legend(loc=legend_positions[index % len(legend_positions)], prop={'size': 6})
        plt.savefig(os.path.join(self.visualization_dir, f'{self.args.model}.pdf'))

    def accumulate_data(self):
        testing_data = []
        for model_name in model_factory.MODEL_ARGUMENTS:
            test_results_path = os.path.join(self.result_dir, model_name, 'testing.csv')
            if os.path.exists(test_results_path):
                testing_table = pd.read_csv(test_results_path)
                testing_data.append(testing_table)
        merged_testing_table = pd.concat(testing_data)
        merged_testing_table.sort_values(merged_testing_table.columns[5], inplace=True)
        merged_testing_table.to_csv(os.path.join(self.result_dir, 'merged_results.csv'), index=False)
        val_loss_data = []
        val_loss_column = ''
        for model_name in model_factory.MODEL_ARGUMENTS:
            training_results_path = os.path.join(self.result_dir, model_name, 'training.csv')
            if os.path.exists(training_results_path):
                training_table = pd.read_csv(training_results_path)
                val_loss_column = training_table.columns[1] if self.args.metric_name == '' else training_table.columns[2]
                val_loss_data.append((model_name, training_table[val_loss_column].tolist()))
        max_len = max([len(x[1]) for x in val_loss_data])
        x_data = np.array(range(1, max_len + 1))
        fig, axis = plt.subplots()
        axis.set_title(f'{val_loss_column} @ {self.__class__.__name__}')
        axis.set_xlabel('epochs')
        axis.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        for model_name, val_losses in val_loss_data:
            axis.plot(x_data[:len(val_losses)], val_losses, label=model_name)
        axis.legend(loc='upper right', prop={'size': 6})
        plt.savefig(os.path.join(self.visualization_dir, 'merged_visualizations.pdf'))

    def train_and_test(self):
        self.check_directories()
        model_name = self.args.model
        assert model_name in model_factory.MODEL_ARGUMENTS
        model_save_location = os.path.join(self.saved_model_dir, model_name)
        tensorboard_save_location = os.path.join(self.tensorboard_dir, model_name)
        if self.args.use_saved_model:
            model = tf.keras.models.load_model(model_save_location)
        else:
            inputs_slice = slice(None) if self.args.use_time_input or len(self.inputs) == 1 else slice(-1)
            model = tf.keras.Model(inputs=self.inputs,
                                   outputs=model_factory.get_model_output_by_name(self.args.model, self.output_size, self.inputs[inputs_slice]),
                                   name=model_name)
            optimizer = tf.keras.optimizers.get({'class_name': self.args.optimizer_name,
                                                 'config': {'learning_rate': self.args.learning_rate}})
            loss = tf.keras.losses.get({'class_name': self.args.loss_name,
                                        'config': self.args.loss_config})
            if self.args.metric_name == '':
                metric = None
            else:
                metric = tf.keras.metrics.get(self.args.metric_name)
            model.compile(optimizer=optimizer, loss=loss, metrics=metric, run_eagerly=self.args.debug)
        model.summary()
        if self.args.debug:
            sample_output = model.predict(tuple((x[:self.args.batch_size] for x in self.training_input_data)))
            sample_loss = model.loss(tuple((x[:self.args.batch_size] for x in self.training_output_data)), sample_output).numpy()
            assert not tf.math.is_nan(sample_loss)
        training_start = time.time()
        fit_result = model.fit(
            x=self.training_input_data,
            y=self.training_output_data,
            batch_size=self.args.batch_size,
            epochs=self.args.epochs,
            validation_data=(self.validation_input_data, self.validation_output_data),
            callbacks=(tf.keras.callbacks.ModelCheckpoint(model_save_location, save_best_only=True),
                       tf.keras.callbacks.EarlyStopping(patience=self.args.no_improvement_abort_patience),
                       tf.keras.callbacks.TerminateOnNaN(),
                       tf.keras.callbacks.ReduceLROnPlateau(patience=self.args.no_improvement_lr_patience),
                       tf.keras.callbacks.TensorBoard(log_dir=tensorboard_save_location)))
        training_end = time.time()
        training_duration = training_end - training_start
        model = tf.keras.models.load_model(model_save_location)
        evaluate_result = model.evaluate(
            x=self.test_input_data,
            y=self.test_output_data,
            batch_size=self.args.batch_size,
            callbacks=(tf.keras.callbacks.TensorBoard(log_dir=tensorboard_save_location)),
            return_dict=True)
        fit_table, evaluate_table = self.create_and_save_tables(model, fit_result, evaluate_result, training_duration)
        self.create_visualization(fit_table, evaluate_table)
        self.accumulate_data()
