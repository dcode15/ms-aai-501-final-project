from nni.experiment import Experiment

experiment = Experiment("local")
experiment.config.trial_command = "python recurrent_neural_network.py"
experiment.config.trial_code_directory = "."
experiment.config.max_experiment_duration = "2h"
# experiment.config.max_trial_number = 10
experiment.config.tuner.name = "Hyperband"
experiment.config.tuner.class_args["optimize_mode"] = "minimize"
experiment.config.trial_concurrency = 1

search_space = {
    "learning_rate": {"_type": "loguniform", "_value": [0.0001, 0.01]},
    "num_hidden_layers": {"_type": "randint", "_value": [1, 20]},
    "lstm_size": {"_type": "choice", "_value": [32, 64, 128, 256]},
    "hidden_layer_size": {"_type": "choice", "_value": [32, 64, 128, 256, 512]},
    "dropout": {"_type": "choice", "_value": [0, 0.25, 0.5, 0.75]},
    "num_epochs": {"_type": "randint", "_value": [1, 5]},
    "weight_decay": {"_type": "choice", "_value": [1e-5, 1e-4, 1e-3, 1e-2]}
}
experiment.config.search_space = search_space

experiment.run(8080)
