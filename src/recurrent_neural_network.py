import nni
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from enums import TextNormalizationStrategy, TokenizationStrategy
from get_logger import logger
from preprocessor import Preprocessor
from rnn_model import RNNModel
from vectorizer import Vectorizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Running with {device} device.")

original_data_path = "../data/Software_5-core.json"

logger.info(f"Reading data from {original_data_path}")
reviews = pd.read_json(original_data_path, lines=True)
reviews = Preprocessor.preprocess_reviews(reviews, tokenization_strategy=TokenizationStrategy.WORD,
                                          text_normalization_strategy=TextNormalizationStrategy.LEMMATIZATION)

review_vectors = Vectorizer.get_word2vec_embeddings(reviews["cleanedReviewText"])
x_data = reviews[["verified", "reviewLengthStd", "reviewAgeStd"]]
y_data = reviews["voteStd"]

review_vectors_train, review_vectors_test, y_train, y_test = train_test_split(review_vectors, y_data, test_size=0.25,
                                                                              random_state=1)
x_train, x_test, _, _ = train_test_split(x_data, y_data, test_size=0.25, random_state=1)

optimized_params = nni.get_next_parameter()
if optimized_params:
    hyperparams = optimized_params
else:
    hyperparams = {
        "dropout": 0.005,
        "lstm_size": 64,
        "hidden_layer_size": 32,
        "learning_rate": 0.0005,
        "num_hidden_layers": 1,
        "num_epochs": 3,
        "weight_decay": 1e-3,
        "num_lstm_layers": 2
    }

model = RNNModel()
model.train(review_vectors_train, x_train, y_train,
            config=hyperparams, num_epochs=hyperparams["num_epochs"])
mse, _, predictions = model.test(review_vectors_test, x_test, y_test)

top_reviews, bottom_reviews = model.get_top_bottom_results(reviews, review_vectors_test, x_test, y_test)
print("Top reviews:")
for review in top_reviews:
    print("--------------------------------------------------------------\n")
    print(f"{review}\n\n")

print("Bottom reviews:")
for review in bottom_reviews:
    print("--------------------------------------------------------------\n")
    print(f"{review}\n\n")

nni.report_final_result(mse)
