from time import process_time

import pandas as pd
from scipy.sparse import hstack
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from PreProcessor import PreProcessor

print("Loading data.")
data_file_path: str = "../data/Software_5-core.json"
reviews = pd.read_json(data_file_path, lines=True)

print("Cleaning data.")
reviews = PreProcessor.clean_review_objects(reviews)
reviews["cleanedTokens"] = reviews["reviewText"].apply(PreProcessor.clean_text).apply(PreProcessor.stem_words)
reviews["cleanedReview"] = reviews["cleanedTokens"].apply(lambda tokens: " ".join(tokens))

print("Vectorizing cleaned reviews.")
tfidf_matrix = PreProcessor.get_tf_idf_vectorization(reviews["cleanedReview"])

print("Training model")
X = hstack([tfidf_matrix, reviews["verified"].values[:, None], reviews["reviewLength"].values[:, None]])
y = reviews["vote"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
gbr = GradientBoostingRegressor(random_state=1)
fit_start = process_time()
gbr.fit(X_train, y_train)
fit_end = process_time()
print(f"Fit time: {fit_end - fit_start}s")

print("Testing model")
pred_start = process_time()
y_pred = gbr.predict(X_test)
pred_end = process_time()
print(f"Prediction time: {pred_end - pred_start}s")

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R-Squared: {r2}")
