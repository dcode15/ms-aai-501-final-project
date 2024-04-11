from time import process_time
import pandas as pd
import numpy as np
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import tensorflow as tf

from enums import TextNormalizationStrategy
from get_logger import logger
from preprocessor import Preprocessor
from vectorizer import Vectorizer

# Define rf_model as an instance of RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=10, min_samples_leaf=10, random_state=1)

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

reviews = pd.read_json('C:/USD/ms-aai-501-final-project/data/Software_5-core.json', lines=True)
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

# Fit the model on your training data
rf_model.fit(X_train, y_train)

# Make predictions with the model
y_predictions = rf_model.predict(X_test)

# Plot Actual vs. Predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_predictions, alpha=0.5)
plt.title('Actual vs. Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.show()

# Calculate and plot feature importances
# Here you may need to adjust the feature names based on your dataset
feature_names = np.append(
    ['TFIDF_Feature_' + str(i) for i in range(review_vectors.shape[1])],
    ['Verified', 'ReviewAgeStd', 'ReviewLengthStd']
)
importances = rf_model.feature_importances_
sorted_indices = np.argsort(importances)[::-1]

top_n = 20  # for example, to show top 20 features
sorted_idx = importances.argsort()[-top_n:]
plt.figure(figsize=(10, 5))
plt.title('Feature Importances')
plt.bar(range(top_n), importances[sorted_idx], align='center')
plt.xticks(range(top_n), np.array(feature_names)[sorted_idx], rotation=90)
plt.tight_layout()
plt.show()