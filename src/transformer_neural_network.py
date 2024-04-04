import nni
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from get_logger import logger
from preprocessor import Preprocessor
from transformer_model import TransformerModel

"""
Trains and evaluates a transformer model on review data using a pretrained DistilRoBERTa model. If called during an 
NNI experiment, will report results to NNI and use provided hyperparameters.
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Running with {device} device.")

data_path = "../data/Software_5-core.json"
logger.info(f"Reading data from {data_path}")
reviews = pd.read_json(data_path, lines=True)
reviews = Preprocessor.clean_review_objects(reviews)
reviews = Preprocessor.standardize_columns(reviews, ["vote", "reviewLength", "reviewAge"])

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
        "learning_rate": 0.0005,
        "hidden_layer_size": 128,
        "num_hidden_layers": 11,
        "num_epochs": 3,
        "weight_decay": 0
    }

model = TransformerModel()
model.train(reviews_train.copy(), x_train, y_train,
            config=hyperparams, num_epochs=hyperparams["num_epochs"])
mse, _, predictions = model.test(reviews_test.copy(), x_test, y_test)

top_reviews, bottom_reviews = model.get_top_bottom_results(reviews_test, x_test, y_test)
print("Top reviews:")
for review in top_reviews:
    print("--------------------------------------------------------------\n")
    print(f"{review}\n\n")

print("Bottom reviews:")
for review in bottom_reviews:
    print("--------------------------------------------------------------\n")
    print(f"{review}\n\n")

nni.report_final_result(mse)
