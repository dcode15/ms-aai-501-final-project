from typing import Tuple, List

import nni
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error

from get_logger import logger


class RNNModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.criterion = None
        self.optimizer = None

    def train(self, review_vectors, x_data, y_data, config, num_epochs=10, batch_size=128) -> None:
        logger.info("Training RNN model.")
        self.model = RNNModule(review_vectors.shape[2], x_data.shape[1], config).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["learning_rate"],
                                          weight_decay=config["weight_decay"])

        for epoch in range(num_epochs):
            for i in range(0, len(review_vectors), batch_size):
                batch_review_vectors = review_vectors[i:i + batch_size]
                batch_x_data = x_data[i:i + batch_size]
                batch_y_data = y_data[i:i + batch_size]

                batch_review_vectors = torch.tensor(batch_review_vectors, dtype=torch.float32).to(self.device)
                batch_x_data = torch.tensor(batch_x_data.values, dtype=torch.float32).to(self.device)
                batch_y_data = torch.tensor(batch_y_data.values, dtype=torch.float32).to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_review_vectors, batch_x_data)
                loss = self.criterion(outputs, batch_y_data)
                loss.backward()
                self.optimizer.step()

            logger.info(f"Epoch {epoch + 1}/{num_epochs}: Loss = {loss.item():.4f}")
            nni.report_intermediate_result(loss.item())

    def test(self, review_vectors, x_data, y_data, batch_size=128) -> Tuple[float, float, List]:
        logger.info("Testing RNN model.")
        review_vectors_test_tensor = torch.tensor(review_vectors, dtype=torch.float32)
        x_test_tensor = torch.tensor(x_data.values, dtype=torch.float32)

        predictions = []
        with torch.no_grad():
            for i in range(0, len(review_vectors_test_tensor), batch_size):
                batch_review_vectors_test = review_vectors_test_tensor[i:i + batch_size].to(self.device)
                batch_x_test = x_test_tensor[i:i + batch_size].to(self.device)

                batch_predictions = self.model(batch_review_vectors_test, batch_x_test)
                predictions.append(batch_predictions.cpu())

        predictions = torch.cat([prediction.flatten() for prediction in predictions]).numpy()

        mse = mean_squared_error(y_data[:len(predictions)], predictions)
        mae = mean_absolute_error(y_data[:len(predictions)], predictions)

        logger.info(f"MSE: {mse}")
        logger.info(f"MAE: {mae}")

        return mse, mae, predictions

    def get_top_bottom_results(self, reviews, review_vectors, x_data, y_data, result_count=3) -> Tuple[
        List[str], List[str]]:
        x_data["reviewAgeStd"] = 0
        _, _, predictions = self.test(review_vectors, x_data, y_data)
        reviews_with_predictions = pd.DataFrame({
            "reviewText": reviews.iloc[x_data.index]["reviewText"],
            "voteStd": predictions
        })
        reviews_with_predictions = reviews_with_predictions.sort_values("voteStd", ascending=False)

        return reviews_with_predictions["reviewText"].head(result_count).values, reviews_with_predictions[
            "reviewText"].tail(result_count).values,


class RNNModule(nn.Module):
    def __init__(self, vector_size, num_other_features, config):
        super(RNNModule, self).__init__()

        num_lstm_layers = config.get("num_lstm_layers", 1)
        num_hidden_layers = config.get("num_hidden_layers", 1)
        hidden_layer_size = config.get("hidden_layer_size", 64)
        dropout_rate = config.get("dropout", 0.5)

        self.lstm_layers = nn.LSTM(vector_size, config["lstm_size"], num_layers=num_lstm_layers, batch_first=True)

        layers = [nn.Linear(config["lstm_size"] + num_other_features, hidden_layer_size), nn.ReLU(),
                  nn.Dropout(dropout_rate)]

        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU(), nn.Dropout(dropout_rate)]

        self.fully_connected_layers = nn.Sequential(*layers)

        self.output_layer = nn.Linear(hidden_layer_size, 1)

    def forward(self, text_input, other_features_input) -> nn.Linear:
        _, (lstm_hidden_state, _) = self.lstm_layers(text_input)
        lstm_output = lstm_hidden_state[-1]
        fully_connected_layers_input = torch.cat((lstm_output, other_features_input), dim=1)
        fully_connected_layers_sequence = self.fully_connected_layers(fully_connected_layers_input)
        return self.output_layer(fully_connected_layers_sequence).squeeze()
