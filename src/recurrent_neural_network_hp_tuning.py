import keras_tuner
from keras.callbacks import EarlyStopping
from keras.layers import Input, LSTM, Dense, Concatenate, Dropout
from keras.models import Model
from sklearn.model_selection import train_test_split

from get_logger import logger
from preprocessor import Preprocessor

review_vectors, x, y = Preprocessor.preprocess_reviews("properties.properties")


def build_model(hp):
    logger.info("Building model.")
    max_sequence_length = review_vectors.shape[1]
    vector_size = review_vectors.shape[2]

    text_input = Input(shape=(max_sequence_length, vector_size), name='text_input')
    lstm_units = hp.Int('lstm_units', min_value=32, max_value=256, step=32)
    lstm_dropout = hp.Float("lstm_dropout", min_value=0, max_value=0.5, step=0.05)
    lstm_output = LSTM(lstm_units, dropout=lstm_dropout, recurrent_dropout=lstm_dropout)(text_input)

    other_features_input = Input(shape=(x.shape[1],), name='other_features_input')
    concatenated = Concatenate()([lstm_output, other_features_input])

    fc_dropout = hp.Float("fc_dropout", min_value=0, max_value=0.5, step=0.05)
    fc_layer_input = Dropout(fc_dropout)(concatenated)
    fc_units = hp.Int('fc_units', min_value=32, max_value=256, step=32)
    fc_activation = hp.Choice("activation", ["relu", "tanh"])
    fc_layer = Dense(fc_units, activation=fc_activation)(fc_layer_input)
    output_layer = Dense(1)(fc_layer)

    model = Model(inputs=[text_input, other_features_input], outputs=output_layer)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="val_loss",
    max_trials=10,
    executions_per_trial=1,
    overwrite=True,
    directory="out"
)
tuner.search_space_summary()

logger.info("Searching hyperparameters.")
early_stopping = EarlyStopping(verbose=1, mode='min')
review_vectors_train, review_vectors_test, y_train, y_test = train_test_split(review_vectors, y, test_size=0.25,
                                                                              random_state=1)
x_train, x_test, _, _ = train_test_split(x, y, test_size=0.25, random_state=1)

tuner.search([review_vectors_train, x_train], y_train,
             validation_data=([review_vectors_test, x_test], y_test),
             batch_size=32,
             epochs=10,
             callbacks=[early_stopping])

logger.info("Best model:")
tuner.get_best_models(num_models=1)[0].summary()

logger.info("Tuner results:")
tuner.results_summary()
