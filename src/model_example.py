from time import process_time

import pandas as pd
from scipy.sparse import hstack
from sklearn.ensemble import GradientBoostingRegressor
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
gbr = GradientBoostingRegressor(random_state=1)
fit_start = process_time()
gbr.fit(X_train, y_train)
fit_end = process_time()
logger.info(f"Fit time: {fit_end - fit_start}s")

logger.info("Testing model")
pred_start = process_time()
y_pred = gbr.predict(X_test)
pred_end = process_time()
logger.info(f"Prediction time: {pred_end - pred_start}s")

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
logger.info(f"MSE: {mse}")
logger.info(f"MAE: {mae}")
