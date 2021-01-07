instructions to run the code:

- make sure you have all dependencies listed in requirements.txt installed
- make sure your current working directory is LongTimeDependenciesLearning
- print the usage of a benchmark script with the command ``python3 -m experiments.benchmarks.{BENCHMARK_NAME} --h``
- possible values for *BENCHMARK_NAME* are: "activity_benchmark", "cell_benchmark", "memory_benchmark", "mnist_benchmark", "walker_benchmark", "xor_benchmark"
- to start a benchmark with a specific model execute ``python3 -m experiments.benchmarks.{BENCHMARK_NAME} --model {MODEL_NAME}``
- possible values for *MODEL_NAME* are: "memory_cell", "memory_layer", "lstm", "differentiable_neural_computer", "unitary_rnn", "enhanced_unitary_rnn", "transformer", "memory_layer_transformer", "
  recurrent_transformer", "gru", "neural_circuit_policies", "ct_rnn", "ct_gru", "ode_lstm"
- please mind that MODEL_NAME "memory_cell" can only be used with the BENCHMARK_NAME "cell_benchmark" and vice versa
