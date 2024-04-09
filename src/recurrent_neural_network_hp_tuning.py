from nni.experiment import Experiment

"""
Performs hyperparameter tuning on the RNN model using the recurrent_neural_network.py script.
"""

experiment = Experiment("local")
experiment.config.trial_command = "python recurrent_neural_network.py"
experiment.config.trial_code_directory = "."
experiment.config.max_experiment_duration = "4h"
# experiment.config.max_trial_number = 10
experiment.config.tuner.name = "TPE"
experiment.config.tuner.class_args["optimize_mode"] = "minimize"
experiment.config.trial_concurrency = 1

search_space = {
    "learning_rate": {"_type": "loguniform", "_value": [0.0001, 0.001]},
    "num_hidden_layers": {"_type": "randint", "_value": [1, 10]},
    "lstm_size": {"_type": "choice", "_value": [32, 64]},
    "hidden_layer_size": {"_type": "choice", "_value": [64, 128, 256]},
    "dropout": {"_type": "choice", "_value": [0, 0.25, 0.5]},
    "num_epochs": {"_type": "randint", "_value": [3, 10]},
    "weight_decay": {"_type": "choice", "_value": [1e-5, 1e-4, 1e-3, 1e-2]},
    "num_lstm_layers": {"_type": "choice", "_value": [1, 2]},
}
experiment.config.search_space = search_space

experiment.run(8080)
