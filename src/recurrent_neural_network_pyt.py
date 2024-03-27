import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from enums import TextNormalizationStrategy, TokenizationStrategy
from get_logger import logger
from preprocessor import Preprocessor
from vectorizer import Vectorizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Running with {device} device.")

reviews = pd.read_json("../data/Software_5-core.json", lines=True)
reviews = Preprocessor.preprocess_reviews(reviews, tokenization_strategy=TokenizationStrategy.WORD,
                                          text_normalization_strategy=TextNormalizationStrategy.LEMMATIZATION)

review_vectors = Vectorizer.get_word2vec_embeddings(reviews["cleanedReviewText"])
x_data = reviews[["verified", "reviewLengthStd", "reviewAgeStd"]]
y_data = reviews["voteStd"]

logger.info("Building model.")
max_sequence_length = review_vectors.shape[1]
vector_size = review_vectors.shape[2]


class RNNModel(nn.Module):
    def __init__(self, vector_size, num_other_features):
        super(RNNModel, self).__init__()
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


model = RNNModel(vector_size, x_data.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

logger.info("Training model.")
review_vectors_train, review_vectors_test, y_train, y_test = train_test_split(review_vectors, y_data, test_size=0.25,
                                                                              random_state=1)
x_train, x_test, _, _ = train_test_split(x_data, y_data, test_size=0.25, random_state=1)

num_epochs = 10
batch_size = 32
for epoch in range(num_epochs):
    for i in range(0, len(review_vectors_train), batch_size):
        batch_review_vectors = review_vectors_train[i:i + batch_size]
        batch_x_data = x_train[i:i + batch_size]
        batch_y_data = y_train[i:i + batch_size]

        batch_review_vectors = torch.tensor(batch_review_vectors, dtype=torch.float32).to(device)
        batch_x_data = torch.tensor(batch_x_data.values, dtype=torch.float32).to(device)
        batch_y_data = torch.tensor(batch_y_data.values, dtype=torch.float32).to(device)

        optimizer.zero_grad()
        outputs = model(batch_review_vectors, batch_x_data)
        loss = criterion(outputs, batch_y_data)
        loss.backward()
        optimizer.step()

    logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

logger.info("Testing model.")
review_vectors_test = torch.tensor(review_vectors_test, dtype=torch.float32).to(device)
x_test = torch.tensor(x_test.values, dtype=torch.float32).to(device)
with torch.no_grad():
    predictions = model(review_vectors_test, x_test)

mse = mean_squared_error(y_test, predictions.cpu().numpy())
mae = mean_absolute_error(y_test, predictions.cpu().numpy())
logger.info(f"MSE: {mse}")
logger.info(f"MAE: {mae}")
