"""
each invocation of run_all_benchmarks_and_models.py generates these four folders in the project directory (assuming default arguments are used): results, saved_models, tensorboard and visualizations
place these folders in a new folder called benchmark_logs/run_{i} where i is the run index after all folders have been manually validated
then execute this script to generate statistics based on all available runs in folder benchmark_logs/statistics
"""

import argparse
import os

import numpy as np
import pandas as pd

import experiments.benchmarks.benchmark as benchmark

parser = argparse.ArgumentParser()
parser.add_argument('--result_folder_name', default='results', type=str)
parser.add_argument('--decimal_places', default=3, type=int)
args = parser.parse_args()

statistics_folder_path = os.path.join('benchmark_logs', 'statistics')
os.makedirs(statistics_folder_path, exist_ok=True)

available_runs = 0
while os.path.exists(os.path.join('benchmark_logs', f'run_{available_runs}')):
    available_runs += 1

for benchmark_name in benchmark.BENCHMARK_NAMES:
    result_tables = []
    header = None
    first_columns = None
    for run_index in range(available_runs):
        result_table = pd.read_csv(os.path.join('benchmark_logs', f'run_{run_index}', args.result_folder_name, benchmark_name, 'merged_results.csv'))
        result_table.sort_values(result_table.columns[0], inplace=True)
        current_header = result_table.columns.values.astype(str)
        assert header is None or np.all(header == current_header)
        header = current_header
        data = result_table.values
        current_first_columns = data[:, :2].astype(str)
        assert first_columns is None or np.all(first_columns == current_first_columns)
        first_columns = current_first_columns
        result_tables.append(data[:, 2:].astype(np.float64))
    assert result_tables and header is not None and first_columns is not None
    merged_results = np.stack(result_tables, -1)
    if merged_results.shape[1] > 4:
        loss_column = -2
    else:
        loss_column = -1
    means = np.mean(merged_results, -1)
    sort_order = means[:, loss_column].argsort()
    sorted_means = np.round(means[sort_order], decimals=args.decimal_places)
    sorted_first_columns = first_columns[sort_order]
    formatted_means = np.char.mod(f'%.{args.decimal_places}f', sorted_means)
    stds = np.std(merged_results, -1, ddof=1)
    sorted_stds = np.round(stds[sort_order], decimals=args.decimal_places)
    formatted_stds = np.char.mod(f'%.{args.decimal_places}f', sorted_stds)
    merged_results_string = np.char.add(np.char.add(formatted_means, ' \u00b1 '), formatted_stds)
    result_table = pd.DataFrame(np.concatenate((sorted_first_columns, merged_results_string), -1), columns=header)
    result_table.to_csv(os.path.join(statistics_folder_path, f'{benchmark_name}.csv'), index=False)
