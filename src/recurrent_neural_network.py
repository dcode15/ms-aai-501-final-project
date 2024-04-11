import os
from time import perf_counter

import nni
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from enums import TextNormalizationStrategy, TokenizationStrategy
from get_logger import logger
from model_analyzer import ModelAnalyzer
from preprocessor import Preprocessor
from rnn_model import RNNModel

"""
Trains and evaluates an RNN model on review data using Word2Vec embeddings. If called during an NNI experiment, will 
report results to NNI and use provided hyperparameters.
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Running with {device} device.")

original_data_path = "../data/Software_5-core.json"
preprocessed_data_path = "../data/Software-Preprocessed-RNN.json"

if os.path.isfile(preprocessed_data_path):
    logger.info(f"Reading data from {preprocessed_data_path}")
    reviews = pd.read_json(preprocessed_data_path, lines=True)
else:
    logger.info(f"Reading data from {original_data_path}")
    reviews = pd.read_json(original_data_path, lines=True)
    reviews = Preprocessor.preprocess_reviews(reviews, tokenization_strategy=TokenizationStrategy.WORD,
                                              text_normalization_strategy=TextNormalizationStrategy.LEMMATIZATION)
    reviews[["reviewText", "cleanedReviewText", "verified", "reviewLengthStd", "reviewAgeStd", "voteStd"]].to_json(
        preprocessed_data_path, lines=True, orient="records")

x_data = reviews[["verified", "reviewLengthStd", "reviewAgeStd"]]
y_data = reviews["voteStd"]

reviews_train, review_vectors_test, y_train, y_test = train_test_split(reviews["cleanedReviewText"], y_data,
                                                                       test_size=0.25,
                                                                       random_state=1)
x_train, x_test, _, _ = train_test_split(x_data, y_data, test_size=0.25, random_state=1)

optimized_params = nni.get_next_parameter()
if optimized_params:
    hyperparams = optimized_params
else:
    hyperparams = {
        "dropout": 0,
        "lstm_size": 64,
        "hidden_layer_size": 64,
        "learning_rate": 0.0002,
        "num_hidden_layers": 4,
        "num_epochs": 4,
        "weight_decay": 0.00001,
        "num_lstm_layers": 1
    }

model = RNNModel()

training_start_time = perf_counter()
model.train(reviews_train, x_train, y_train,
            config=hyperparams, num_epochs=hyperparams["num_epochs"], validation_split=0)
training_end_time = perf_counter()
logger.info(f"Training time: {round(training_end_time - training_start_time)}s")

inference_start_time = perf_counter()
predictions = model.test(review_vectors_test, x_test)
inference_end_time = perf_counter()
logger.info(f"Inference time: {round(inference_end_time - inference_start_time)}s")

metrics = ModelAnalyzer.get_key_metrics(y_test, predictions)
logger.info(f"Model metrics: {metrics}")

ModelAnalyzer.get_top_bottom_results(reviews, x_test, predictions, print_reviews=True)
ModelAnalyzer.plot_predictions_vs_actuals(y_test, predictions)

nni.report_final_result(metrics["mse"])
