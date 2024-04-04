from nni.experiment import Experiment

"""
Performs hyperparameter tuning on the transformer model using the transformer_neural_network.py script.
"""

experiment = Experiment("local")
experiment.config.trial_command = "python transformer_neural_network.py"
experiment.config.trial_code_directory = "."
experiment.config.max_experiment_duration = "3h"
# experiment.config.max_trial_number = 10
experiment.config.tuner.name = "Metis"
experiment.config.tuner.class_args["optimize_mode"] = "minimize"
experiment.config.trial_concurrency = 1

search_space = {
    "learning_rate": {"_type": "choice", "_value": [1e-5, 1e-4, 1e-3, 1e-2]},
    "num_hidden_layers": {"_type": "randint", "_value": [1, 20]},
    "hidden_layer_size": {"_type": "choice", "_value": [32, 64, 128, 256, 512]},
    "dropout": {"_type": "choice", "_value": [0, 0.25, 0.5, 0.75]},
    "num_epochs": {"_type": "randint", "_value": [1, 10]},
    "weight_decay": {"_type": "choice", "_value": [0, 1e-5, 1e-4, 1e-3, 1e-2]},
}
experiment.config.search_space = search_space

experiment.run(8080)
