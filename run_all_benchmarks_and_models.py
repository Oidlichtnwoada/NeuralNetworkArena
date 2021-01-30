import os
import subprocess

BENCHMARK_NAMES = ['cell', 'activity', 'add', 'memory', 'mnist', 'walker']

MODEL_NAMES = ['lstm', 'differentiable_neural_computer', 'unitary_rnn', 'matrix_exponential_unitary_rnn',
               'transformer', 'recurrent_network_augmented_transformer', 'gru', 'ct_gru', 'memory_cell',
               'ode_lstm', 'unitary_ncp', 'neural_circuit_policies', 'ct_rnn', 'memory_augmented_transformer']

for benchmark_name in BENCHMARK_NAMES:
    for model_name in MODEL_NAMES:
        if not os.path.exists(os.path.join(os.path.curdir, 'results', benchmark_name, model_name, 'training.csv')):
            if benchmark_name == 'cell' and model_name != 'memory_cell':
                continue
            if benchmark_name != 'cell' and model_name == 'memory_cell':
                continue
            subprocess.run(f'py -m experiments.benchmarks.{benchmark_name}_benchmark --model {model_name}')
