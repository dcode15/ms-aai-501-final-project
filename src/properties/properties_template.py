from properties.enums import TokenizationType, TextNormalizationStrategy, VectorizationStrategy

data_file_path = "../data/Software_5-core.json"
lowercase_text = True
remove_punctuation = True
remove_stopwords = True
tokenization_type = TokenizationType.WORD
text_normalization_strategy = TextNormalizationStrategy.LEMMATIZATION
vectorization_strategy = VectorizationStrategy.TF_IDF
training_features = ["verified", "reviewLength", "reviewAge"]
