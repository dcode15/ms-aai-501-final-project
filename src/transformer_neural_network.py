import os
from time import process_time

import nni
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from get_logger import logger
from model_analyzer import ModelAnalyzer
from preprocessor import Preprocessor
from transformer_model import TransformerModel

"""
Trains and evaluates a transformer model on review data using a pretrained DistilRoBERTa model. If called during an 
NNI experiment, will report results to NNI and use provided hyperparameters.
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Running with {device} device.")

original_data_path = "../data/Software_5-core.json"
preprocessed_data_path = "../data/Software-Preprocessed-Transformer.json"

if os.path.isfile(preprocessed_data_path):
    logger.info(f"Reading data from {preprocessed_data_path}")
    reviews = pd.read_json(preprocessed_data_path, lines=True)
else:
    logger.info(f"Reading data from {original_data_path}")
    reviews = pd.read_json(original_data_path, lines=True)
    reviews = Preprocessor.clean_review_objects(reviews)
    reviews = Preprocessor.standardize_columns(reviews, ["vote", "reviewLength", "reviewAge"])
    reviews[["reviewText", "verified", "reviewLengthStd", "reviewAgeStd", "voteStd"]].to_json(
        preprocessed_data_path, lines=True, orient="records")

x_data = reviews[["verified", "reviewLengthStd", "reviewAgeStd"]]
y_data = reviews["voteStd"]

reviews_train, reviews_test, y_train, y_test = train_test_split(reviews["reviewText"], y_data, test_size=0.25,
                                                                random_state=1)
x_train, x_test, _, _ = train_test_split(x_data, y_data, test_size=0.25, random_state=1)

optimized_params = nni.get_next_parameter()
if optimized_params:
    hyperparams = optimized_params
else:
    hyperparams = {
        "dropout": 0,
        "learning_rate": 0.0025,
        "hidden_layer_size": 64,
        "num_hidden_layers": 13,
        "num_epochs": 2,
        "weight_decay": 0
    }

model = TransformerModel()

training_start_time = process_time()
model.train(reviews_train.copy(), x_train, y_train,
            config=hyperparams, num_epochs=hyperparams["num_epochs"])
training_end_time = process_time()
logger.info(f"Training time: {training_end_time - training_start_time}s")

inference_start_time = process_time()
predictions = model.test(reviews_test.copy(), x_test)
inference_end_time = process_time()
logger.info(f"Inference time: {inference_end_time - inference_start_time}s")

ModelAnalyzer.plot_predictions_vs_actuals(y_test, predictions)
metrics = ModelAnalyzer.get_key_metrics(y_test, predictions)
logger.info(f"Model metrics: {metrics}")

ModelAnalyzer.get_top_bottom_results(reviews, x_test, predictions, print_reviews=True)

nni.report_final_result(metrics["mse"])
