from math import ceil, sqrt
from typing import Tuple, List

import nni
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import kendalltau
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from transformers import RobertaForSequenceClassification, RobertaTokenizer

from get_logger import logger


class TransformerModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def train(self, reviews, x_data, y_data, config, num_epochs=10, batch_size=128, validation_split=0.2) -> None:
        logger.info("Training Transformer model.")
        self.model = TransformerModule(x_data.shape[1], config).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["learning_rate"],
                                          weight_decay=config["weight_decay"])
        if validation_split == 0:
            reviews_train = reviews
            x_train = x_data
            y_train = y_data
        else:
            x_train, x_validation, y_train, y_validation, reviews_train, reviews_validation = train_test_split(
                x_data, y_data, reviews, test_size=validation_split, random_state=1)

        for epoch in range(num_epochs):
            self.model.train()
            for i in range(0, len(reviews_train), batch_size):
                if (int(i / batch_size) + 1) % 50 == 0:
                    logger.info(
                        f"Processing batch {int(i / batch_size) + 1} of {ceil(len(reviews_train) / batch_size)}")
                batch_reviews = reviews_train[i:i + batch_size]
                batch_x_data = x_train[i:i + batch_size]
                batch_y_data = y_train[i:i + batch_size]

                batch_reviews = self.tokenizer(list(batch_reviews), padding=True, truncation=True,
                                               return_tensors="pt").to(self.device)
                batch_x_data = torch.tensor(batch_x_data.values, dtype=torch.float32).to(self.device)
                batch_y_data = torch.tensor(batch_y_data.values, dtype=torch.float32).to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_reviews, batch_x_data)
                loss = self.criterion(outputs, batch_y_data)
                loss.backward()
                self.optimizer.step()

            if validation_split == 0:
                logger.info(f"Epoch {epoch + 1}/{num_epochs}: Loss = {round(loss.item(), 3)}")
                nni.report_intermediate_result(loss.item())
            else:
                validation_loss = self.__validate(reviews_validation, x_validation, y_validation, batch_size)
                logger.info(f"Epoch {epoch + 1}/{num_epochs}: Loss = {round(validation_loss, 3)}")
                nni.report_intermediate_result(validation_loss)

    def __validate(self, reviews, x_data, y_data, batch_size=128):
        self.model.eval()
        validation_losses = []
        with torch.no_grad():
            for i in range(0, len(x_data), batch_size):
                batch_reviews = self.tokenizer(list(reviews[i:i + batch_size].values), padding=True, truncation=True,
                                               return_tensors="pt").to(self.device)
                batch_x_data = torch.tensor(x_data[i:i + batch_size].values, dtype=torch.float32).to(self.device)
                batch_y_data = torch.tensor(y_data[i:i + batch_size].values, dtype=torch.float32).to(self.device)

                outputs = self.model(batch_reviews, batch_x_data)
                loss = self.criterion(outputs, batch_y_data)
                validation_losses.append(loss.item())

        avg_validation_loss = sum(validation_losses) / len(validation_losses)
        return avg_validation_loss

    def test(self, reviews, x_data, y_data, batch_size=128) -> Tuple[float, float, float, List]:
        logger.info("Testing Transformer model.")
        reviews_test = self.tokenizer(list(reviews), padding=True, truncation=True, return_tensors="pt")
        x_test_tensor = torch.tensor(x_data.values, dtype=torch.float32)

        predictions = []
        with torch.no_grad():
            for i in range(0, len(reviews_test["input_ids"]), batch_size):
                batch_reviews_test = {k: v[i:i + batch_size].to(self.device) for k, v in reviews_test.items()}
                batch_x_test = x_test_tensor[i:i + batch_size].to(self.device)

                batch_predictions = self.model(batch_reviews_test, batch_x_test)
                predictions.append(batch_predictions.cpu())

        predictions = torch.cat([prediction.flatten() for prediction in predictions]).numpy()

        mse = mean_squared_error(y_data[:len(predictions)], predictions)
        mae = mean_absolute_error(y_data[:len(predictions)], predictions)
        kendalls_tau = kendalltau(y_data[:len(predictions)], predictions)

        logger.info(f"MSE: {mse}")
        logger.info(f"RMSE: {sqrt(mse)}")
        logger.info(f"MAE: {mae}")
        logger.info(f"Kendall's Tau: value = {kendalls_tau.statistic}, p-value = {kendalls_tau.pvalue}")

        return mse, mae, kendalls_tau, predictions

    def get_top_bottom_results(self, reviews, x_data, y_data, result_count=3) -> Tuple[
        List[str], List[str]]:
        logger.info("Getting top and bottom results.")
        x_data["reviewAgeStd"] = 0
        _, _, _, predictions = self.test(reviews.copy(), x_data, y_data)
        reviews_with_predictions = pd.DataFrame({
            "reviewText": reviews,
            "voteStd": predictions
        })
        reviews_with_predictions = reviews_with_predictions.sort_values("voteStd", ascending=False)

        return reviews_with_predictions["reviewText"].head(result_count).values, reviews_with_predictions[
            "reviewText"].tail(result_count).values


class TransformerModule(nn.Module):
    def __init__(self, num_other_features, config):
        super(TransformerModule, self).__init__()
        self.roberta_model = RobertaForSequenceClassification.from_pretrained("../models/fine_tuned_distilroberta",
                                                                              num_labels=1)
        num_hidden_layers = config.get("num_hidden_layers", 1)
        hidden_layer_size = config.get("hidden_layer_size", 64)
        dropout_rate = config.get("dropout", 0.5)

        layers = [nn.Linear(1 + num_other_features, hidden_layer_size), nn.ReLU(),
                  nn.Dropout(dropout_rate)]

        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU(), nn.Dropout(dropout_rate)]

        self.fully_connected_layers = nn.Sequential(*layers)

        self.output_layer = nn.Linear(hidden_layer_size, 1)

        for parameter in self.roberta_model.parameters():
            parameter.requires_grad = False

    def forward(self, text_input, other_features_input) -> nn.Linear:
        roberta_output = self.roberta_model(**text_input).logits
        fully_connected_layers_input = torch.cat((roberta_output, other_features_input), dim=1)
        fully_connected_layers_sequence = self.fully_connected_layers(fully_connected_layers_input)
        return self.output_layer(fully_connected_layers_sequence).squeeze()
