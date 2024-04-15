from typing import List

import torch
from sklearn.linear_model import LinearRegression
from transformers import RobertaForSequenceClassification, RobertaTokenizer

from get_logger import logger

"""
Reusable class representing a model combining a DistilRoBERTa transformer model with a linear regression model. 
"""


class LinRegTransformerModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear_regression_model = None
        self.criterion = None
        self.optimizer = None
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.transformer_model = RobertaForSequenceClassification.from_pretrained(
            "../models/fine_tuned_distilroberta").to(self.device)
        self.transformer_model.eval()

    def train(self, reviews, x_data, y_data) -> None:
        transformer_predictions = self.__get_transformer_predictions(reviews)
        x_data["predictions"] = transformer_predictions
        self.linear_regression_model = LinearRegression()
        self.linear_regression_model.fit(x_data, y_data)

        logger.info("Linear regression coefficients:")
        for feature, coef in zip(x_data.columns, self.linear_regression_model.coef_):
            print(f"Feature: {feature}, Coefficient: {coef}")

    def test(self, reviews, x_data) -> List[float]:
        transformer_predictions = self.__get_transformer_predictions(reviews)
        x_data["predictions"] = transformer_predictions
        return self.linear_regression_model.predict(x_data)

    def __get_transformer_predictions(self, reviews):
        predictions = []
        for index, text in enumerate(reviews):
            if index % 1000 == 0:
                logger.info(f"Processing review {index} of {len(reviews)}")
            inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.transformer_model(**inputs)
                logits = outputs.logits
                regression_output = logits.squeeze().item()
                predictions.append(regression_output)

        return predictions
