from math import sqrt
from typing import Tuple

import pandas as pd
from numpy import ndarray
from sklearn.metrics import mean_squared_error, mean_absolute_error, ndcg_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from get_logger import logger


class ModelAnalyzer:

    @staticmethod
    def get_key_metrics(actual_values, predicted_values, ndcg_k=None):
        """
        Calculates key metrics for assessing model performance.

        :param actual_values: The true values
        :param predicted_values: The predicted values
        :param ndcg_k: The number of positions to consider in the calculation of the Normalized Discounted Cumulative Gain.
                       If None, all positions are considered.
        :returns: A dictionary containing the Mean Squared Error (mse), Root Mean Squared Error (rmse),
                  Mean Absolute Error (mae) and Normalized Discounted Cumulative Gain (ndcg).
        """
        logger.info("Calculating model metrics.")
        mse = mean_squared_error(actual_values, predicted_values)
        mae = mean_absolute_error(actual_values, predicted_values)

        scaler = MinMaxScaler()
        ndcg_actual_values = actual_values.values.reshape(-1, 1)
        ndcg_predicted_values = predicted_values.reshape(-1, 1)
        scaler.fit(ndcg_actual_values)
        ndcg_actual_values = scaler.transform(ndcg_actual_values)
        ndcg_predicted_values = scaler.transform(ndcg_predicted_values)
        ndcg = ndcg_score([ndcg_actual_values.flatten()], [ndcg_predicted_values.flatten()], k=ndcg_k)

        return {
            "mse": mse,
            "rmse": sqrt(mse),
            "mae": mae,
            "ndcg": ndcg
        }

    @staticmethod
    def get_top_bottom_results(reviews, x_test, predictions, result_count=3, print_reviews=False) -> Tuple[
        ndarray, ndarray]:
        """
        This method calculates the top and bottom ranked reviews based on the model predictions.

        Parameters:
        :param reviews: The original dataframe containing the reviewText
        :param x_test: The df used as the input features when testing the model.
        :param predictions: The predictions obtained from the model.
        :param result_count: The number of top and bottom results to return. Default is 3.

        Returns:
        Tuple[List[str], List[str]]: Returns two lists of strings; first being the top 'result_count' reviews,
                                     second being the bottom 'result_count' reviews.
        """
        logger.info("Calculating top and bottom ranked reviews.")
        x_test["reviewAgeStd"] = 0
        reviews_with_predictions = pd.DataFrame({
            "reviewText": reviews.iloc[x_test.index]["reviewText"],
            "voteStd": predictions
        })
        reviews_with_predictions = reviews_with_predictions.sort_values("voteStd", ascending=False)
        top_reviews = reviews_with_predictions["reviewText"].head(result_count).values
        bottom_reviews = reviews_with_predictions["reviewText"].tail(result_count).values

        if print_reviews:
            print("Top reviews:")
            for review in top_reviews:
                print("--------------------------------------------------------------\n")
                print(f"{review}\n\n")

            print("Bottom reviews:")
            for review in bottom_reviews:
                print("--------------------------------------------------------------\n")
                print(f"{review}\n\n")

        return top_reviews, bottom_reviews

    @staticmethod
    def plot_predictions_vs_actuals(actual_values, predicted_values, title="Predictions vs Actuals",
                                    x_label="Actual Values", y_label="Predicted Values"):
        plt.figure(figsize=(10, 6))
        plt.scatter(actual_values, predicted_values, alpha=0.5)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.plot([actual_values.min(), actual_values.max()], [actual_values.min(), actual_values.max()], 'k--', lw=4)
        plt.show()
