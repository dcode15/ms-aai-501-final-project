from keras.layers import Input, LSTM, Dense, Concatenate
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from preprocessor import Preprocessor
from get_logger import logger

review_vectors, x, y = Preprocessor.preprocess_reviews("properties.properties")

max_sequence_length = review_vectors.shape[1]
vector_size = review_vectors.shape[2]

text_input = Input(shape=(max_sequence_length, vector_size), name='text_input')
lstm_output = LSTM(50)(text_input)

other_features_input = Input(shape=(x.shape[1],), name='other_features_input')
concatenated = Concatenate()([lstm_output, other_features_input])

fc_layer = Dense(100, activation='relu')(concatenated)
output_layer = Dense(1)(fc_layer)

logger.info("Building model.")
model = Model(inputs=[text_input, other_features_input], outputs=output_layer)
model.compile(optimizer='adam', loss='mean_squared_error')

logger.info("Training model.")
review_vectors_train, review_vectors_test, y_train, y_test = train_test_split(review_vectors, y, test_size=0.25,
                                                                              random_state=1)
x_train, x_test, _, _ = train_test_split(x, y, test_size=0.25, random_state=1)
model.fit([review_vectors, x], y, batch_size=32, epochs=10, validation_split=0.2)

logger.info("Testing model.")
predictions = model.predict([review_vectors_test, x_test])
mse = mean_squared_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R^2: {r2}")
