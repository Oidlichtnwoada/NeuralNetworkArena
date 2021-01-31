import argparse
import os
import subprocess

import experiments.benchmarks.benchmark as benchmark
import experiments.models.model_factory as model_factory

parser = argparse.ArgumentParser()
parser.add_argument('--cuda_visible_devices', default='', type=str)
parser.add_argument('--result_folder_name', default='results', type=str)
parser.add_argument('--python_executable_name', default='python3.8', type=str)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices

for benchmark_name in benchmark.BENCHMARK_NAMES:
    for model_argument in model_factory.MODEL_ARGUMENTS:
        if not os.path.exists(os.path.join(os.path.curdir, args.result_folder_name, benchmark_name, model_argument, 'training.csv')):
            if benchmark_name == 'cell' and model_argument != 'memory_cell':
                continue
            if benchmark_name != 'cell' and model_argument == 'memory_cell':
                continue
            subprocess.run([f'{args.python_executable_name}', '-m', f'experiments.benchmarks.{benchmark_name}_benchmark',
                            '--model', f'{model_argument}', '--result_folder_name', f'{args.result_folder_name}'])
