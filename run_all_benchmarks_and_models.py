import os
import subprocess

import experiments.benchmarks.benchmark as benchmark
import experiments.models.model_factory as model_factory

for benchmark_name in benchmark.BENCHMARK_NAMES:
    for model_argument in model_factory.MODEL_ARGUMENTS:
        if not os.path.exists(os.path.join(os.path.curdir, 'results', benchmark_name, model_argument, 'training.csv')):
            if benchmark_name == 'cell' and model_argument != 'memory_cell':
                continue
            if benchmark_name != 'cell' and model_argument == 'memory_cell':
                continue
            subprocess.run(f'python3.8 -m experiments.benchmarks.{benchmark_name}_benchmark --model {model_argument}')
