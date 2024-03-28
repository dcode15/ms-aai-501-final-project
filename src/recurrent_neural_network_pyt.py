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

reviews = pd.read_json("../data/Software_5-core.json", lines=True)
reviews = Preprocessor.preprocess_reviews(reviews, tokenization_strategy=TokenizationStrategy.WORD,
                                          text_normalization_strategy=TextNormalizationStrategy.LEMMATIZATION)

review_vectors = Vectorizer.get_word2vec_embeddings(reviews["cleanedReviewText"])
x_data = reviews[["verified", "reviewLengthStd", "reviewAgeStd"]]
y_data = reviews["voteStd"]

review_vectors_train, review_vectors_test, y_train, y_test = train_test_split(review_vectors, y_data, test_size=0.25,
                                                                              random_state=1)
x_train, x_test, _, _ = train_test_split(x_data, y_data, test_size=0.25, random_state=1)

model = RNNModel(review_vectors.shape[2], x_data.shape[1])
model.train(review_vectors_train, x_train, y_train)
model.test(review_vectors_test, x_test, y_test)
