from typing import Tuple

import nni
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from transformers import RobertaForSequenceClassification, RobertaTokenizer

from get_logger import logger


class TransformerModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def train(self, reviews, x_data, y_data, config, num_epochs=10, batch_size=128):
        logger.info("Training Transformer model.")
        self.model = TransformerModule(x_data.shape[1], config).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["learning_rate"],
                                          weight_decay=config["weight_decay"])

        for epoch in range(num_epochs):
            for i in range(0, len(reviews), batch_size):
                batch_reviews = reviews[i:i + batch_size]
                batch_x_data = x_data[i:i + batch_size]
                batch_y_data = y_data[i:i + batch_size]

                batch_reviews = self.tokenizer(list(batch_reviews), padding=True, truncation=True,
                                               return_tensors="pt").to(self.device)
                batch_x_data = torch.tensor(batch_x_data.values, dtype=torch.float32).to(self.device)
                batch_y_data = torch.tensor(batch_y_data.values, dtype=torch.float32).to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_reviews, batch_x_data)
                loss = self.criterion(outputs, batch_y_data)
                loss.backward()
                self.optimizer.step()

            logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
            nni.report_intermediate_result(loss.item())

    def test(self, reviews, x_data, y_data, batch_size=128) -> Tuple:
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

        logger.info(f"MSE: {mse}")
        logger.info(f"MAE: {mae}")

        return mse, mae, predictions

    def get_top_bottom_results(self, reviews, x_data, y_data, result_count=3) -> Tuple:
        logger.info("Getting top and bottom results.")
        x_data["reviewAgeStd"] = 0
        _, _, predictions = self.test(reviews.copy(), x_data, y_data)
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
        self.roberta = RobertaForSequenceClassification.from_pretrained("../models/fine_tuned_distilroberta",
                                                                        num_labels=1)
        num_hidden_layers = config.get("num_hidden_layers", 1)
        hidden_layer_size = config.get("hidden_layer_size", 64)
        dropout_rate = config.get("dropout", 0.5)

        layers = [nn.Linear(1 + num_other_features, hidden_layer_size), nn.ReLU(),
                  nn.Dropout(dropout_rate)]

        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU(), nn.Dropout(dropout_rate)]

        self.hidden_layers = nn.Sequential(*layers)

        self.fc_out = nn.Linear(hidden_layer_size, 1)

        for param in self.roberta.parameters():
            param.requires_grad = False

    def forward(self, text_input, other_features_input):
        roberta_output = self.roberta(**text_input).logits
        concatenated = torch.cat((roberta_output, other_features_input), dim=1)
        fc_output = self.hidden_layers(concatenated)
        return self.fc_out(fc_output).squeeze()
