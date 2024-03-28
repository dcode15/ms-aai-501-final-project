import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error

from get_logger import logger


class RNNModel:
    def __init__(self, vector_size, num_other_features):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RNNModule(vector_size, num_other_features).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def train(self, review_vectors, x_data, y_data, num_epochs=10, batch_size=32):
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

            logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    def test(self, review_vectors, x_data, y_data, batch_size=32):
        review_vectors_test_tensor = torch.tensor(review_vectors, dtype=torch.float32)
        x_test_tensor = torch.tensor(x_data.values, dtype=torch.float32)

        predictions = []
        with torch.no_grad():
            for i in range(0, len(review_vectors_test_tensor), batch_size):
                batch_review_vectors_test = review_vectors_test_tensor[i:i + batch_size].to(self.device)
                batch_x_test = x_test_tensor[i:i + batch_size].to(self.device)

                batch_predictions = self.model(batch_review_vectors_test, batch_x_test)
                predictions.append(batch_predictions.cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)

        mse = mean_squared_error(y_data[:len(predictions)], predictions)
        mae = mean_absolute_error(y_data[:len(predictions)], predictions)

        logger.info(f"MSE: {mse}")
        logger.info(f"MAE: {mae}")


class RNNModule(nn.Module):
    def __init__(self, vector_size, num_other_features):
        super(RNNModule, self).__init__()
        self.lstm = nn.LSTM(vector_size, 50, batch_first=True, dropout=0.25)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(50 + num_other_features, 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 1)

    def forward(self, text_input, other_features_input):
        _, (hidden_state, _) = self.lstm(text_input)
        hidden_state = hidden_state.squeeze(0)
        concatenated = torch.cat((hidden_state, other_features_input), dim=1)
        fc_input = self.dropout(concatenated)
        fc_output = self.relu(self.fc1(fc_input))
        output = self.fc2(fc_output)
        return output
