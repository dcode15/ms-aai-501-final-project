import pandas as pd
from gensim.models import Word2Vec, FastText

from enums import TokenizationStrategy, TextNormalizationStrategy
from get_logger import logger
from preprocessor import Preprocessor

data_path = "../data/Software.json"
logger.info(f"Reading data from {data_path}")
reviews = pd.read_json(data_path, lines=True)

logger.info("Preprocessing reviews.")
reviews = Preprocessor.preprocess_reviews(reviews, tokenization_strategy=TokenizationStrategy.WORD,
                                          text_normalization_strategy=TextNormalizationStrategy.LEMMATIZATION)

logger.info("Training FastText model.")
fasttext_model = FastText(sentences=reviews["cleanedReviewText"], vector_size=100, window=5, min_count=5, workers=4,
                          sg=1)
fasttext_model.save("../models/fasttext/trained_fasttext.model")

logger.info("Training Word2Vec model.")
word2vec_model = Word2Vec(sentences=reviews["cleanedReviewText"], vector_size=100, window=5, min_count=5, workers=4,
                          sg=1)
word2vec_model.save("../models/word2vec/trained_word2vec.model")
