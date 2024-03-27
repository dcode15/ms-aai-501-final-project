from keras.callbacks import EarlyStopping
from keras.layers import Input, LSTM, Dense, Concatenate, Dropout
from keras.models import Model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from get_logger import logger
from preprocessor import Preprocessor

review_vectors, x, y = Preprocessor.preprocess_reviews("properties.properties")

logger.info("Building model.")
max_sequence_length = review_vectors.shape[1]
vector_size = review_vectors.shape[2]

text_input = Input(shape=(max_sequence_length, vector_size), name='text_input')
lstm_output = LSTM(50, dropout=0.25, recurrent_dropout=0.25)(text_input)

other_features_input = Input(shape=(x.shape[1],), name='other_features_input')
concatenated = Concatenate()([lstm_output, other_features_input])

fc_layer_input = Dropout(0.5)(concatenated)
fc_layer = Dense(100, activation='relu')(fc_layer_input)
output_layer = Dense(1)(fc_layer)

logger.info("Building model.")
model = Model(inputs=[text_input, other_features_input], outputs=output_layer)
model.compile(optimizer='adam', loss='mean_squared_error')

logger.info("Training model.")
early_stopping = EarlyStopping(verbose=1, mode='min')
review_vectors_train, review_vectors_test, y_train, y_test = train_test_split(review_vectors, y, test_size=0.25,
                                                                              random_state=1)
x_train, x_test, _, _ = train_test_split(x, y, test_size=0.25, random_state=1)
model.fit([review_vectors_train, x_train], y_train, batch_size=32, epochs=10, validation_split=0.2,
          callbacks=[early_stopping])

logger.info("Testing model.")
predictions = model.predict([review_vectors_test, x_test])

mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
logger.info(f"MSE: {mse}")
logger.info(f"MAE: {mae}")

# GBR best result: 0.220/0.116
# RNN best result: 0.18/0.14
