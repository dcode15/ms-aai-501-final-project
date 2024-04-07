from time import process_time

import pandas as pd
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from enums import TextNormalizationStrategy
from get_logger import logger
from preprocessor import Preprocessor
from vectorizer import Vectorizer

reviews = pd.read_json("../data/Software_5-core.json", lines=True)
reviews = Preprocessor.preprocess_reviews(reviews, text_normalization_strategy=TextNormalizationStrategy.LEMMATIZATION)

review_vectors = Vectorizer.get_tf_idf_vectors(reviews["cleanedReviewText"])
x_data = hstack([review_vectors, reviews["verified"].values[:, None], reviews["reviewAgeStd"].values[:, None],
                 reviews["reviewLengthStd"].values[:, None]])
y_data = reviews["voteStd"]

logger.info("Training model.")
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=1)
gbr_model = RandomForestRegressor(random_state=1)
training_start_time = process_time()
gbr_model.fit(X_train, y_train)
training_end_time = process_time()
logger.info(f"Training time: {training_end_time - training_start_time}s")

logger.info("Testing model")
inference_start_time = process_time()
y_predictions = gbr_model.predict(X_test)
inference_end_time = process_time()
logger.info(f"Inference time: {inference_end_time - inference_start_time}s")

mse = mean_squared_error(y_test, y_predictions)
mae = mean_absolute_error(y_test, y_predictions)
logger.info(f"MSE: {mse}")
logger.info(f"MAE: {mae}")
