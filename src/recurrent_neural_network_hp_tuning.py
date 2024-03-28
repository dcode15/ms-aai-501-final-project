from nni.experiment import Experiment

experiment = Experiment('local')
experiment.config.trial_command = 'python recurrent_neural_network.py'
experiment.config.trial_code_directory = '.'
experiment.config.max_experiment_duration = '8h'
# experiment.config.max_trial_number = 10
experiment.config.tuner.name = 'Random'
experiment.config.tuner.class_args['optimize_mode'] = 'minimize'
experiment.config.trial_concurrency = 1

search_space = {
    'learning_rate': {'_type': 'loguniform', '_value': [0.0001, 0.1]},
    'lstm_size': {'_type': 'choice', '_value': [32, 64, 128, 256, 512]},
    'hidden_layer_size': {'_type': 'choice', '_value': [32, 64, 128, 256, 512]},
    "dropout": {'_type': 'uniform', '_value': [0, 0.5]},
}
experiment.config.search_space = search_space

experiment.run(8080)
