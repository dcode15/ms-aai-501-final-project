import importlib
import string
from typing import List, Any, Tuple

import nltk
import pandas as pd
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from properties.enums import TextNormalizationStrategy, VectorizationStrategy
from get_logger import logger

pd.options.mode.chained_assignment = None


class Preprocessor:
    """
    A class for preprocessing text data.
    """
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("punkt", quiet=True)

    @staticmethod
    def preprocess_reviews(properties_file: str) -> Tuple:
        """
        Preprocesses review data for sentiment analysis or other NLP tasks based on specified properties.
        This method involves several steps:
        1. Loading review data from a JSON file specified in the properties module.
        2. Cleaning and normalizing the review text and votes.
        3. Applying text normalization strategies such as stemming or lemmatization.
        4. Vectorizing the preprocessed review texts based on the specified vectorization strategy (TF-IDF or Word2Vec).

        :param properties_file: The name of the properties file (without the .py)
        :return: Depending on the vectorization strategy specified in the properties, this method returns:
        - For TF-IDF: A tuple (x_data, y_data) where `x_data` is a sparse matrix of TF-IDF vectorized texts concatenated
          with other specified training features, and `y_data` contains the normalized vote scores.
        - For Word2Vec: A tuple (review_vectors, x_data, y_data) where `review_vectors` are the padded sequences of
          Word2Vec vectors for each review, `x_data` contains other specified training features, and `y_data` contains
          the normalized vote scores.
        """
        properties = importlib.import_module(properties_file)

        logger.info("Loading data.")
        reviews = pd.read_json(properties.data_file_path, lines=True)
        logger.info("Cleaning data.")
        reviews = Preprocessor.clean_review_objects(reviews)
        reviews = Preprocessor.standardize_columns(reviews, ["vote", "reviewLength", "reviewAge"])
        reviews["cleanedTokens"] = reviews["reviewText"].apply(Preprocessor.clean_text,
                                                               lowercase_text=properties.lowercase_text,
                                                               remove_punctuation=properties.remove_punctuation,
                                                               remove_stopwords=properties.remove_stopwords)

        if properties.text_normalization_strategy == TextNormalizationStrategy.STEMMING:
            reviews["cleanedTokens"] = reviews["cleanedTokens"].apply(Preprocessor.stem_words)
        elif properties.text_normalization_strategy == TextNormalizationStrategy.LEMMATIZATION:
            reviews["cleanedTokens"] = reviews["cleanedTokens"].apply(Preprocessor.lemmatize_words)

        logger.info("Vectorizing reviews.")
        y_data = reviews["voteStd"]
        if properties.vectorization_strategy is VectorizationStrategy.TF_IDF:
            reviews["cleanedReview"] = reviews["cleanedTokens"].apply(lambda tokens: " ".join(tokens))
            review_vectors = Preprocessor.get_tf_idf_vectorization(reviews["cleanedReview"])
            reviews.drop("cleanedReview", axis=1)
            hstack_args = [reviews[feature].values[:, None] for feature in properties.training_features]
            x_data = hstack([review_vectors, *hstack_args])
            return x_data, y_data
        elif properties.vectorization_strategy is VectorizationStrategy.WORD_2_VEC:
            w2v_model = Word2Vec(sentences=reviews["cleanedTokens"], vector_size=100, window=5, min_count=1, workers=4)
            review_vectors = []
            for reviewTokens in reviews["cleanedTokens"]:
                review_vectors.append([w2v_model.wv[token] for token in reviewTokens if token in w2v_model.wv])
            max_length = max(len(review) for review in review_vectors)
            review_vectors = pad_sequences(review_vectors, maxlen=max_length, padding='post', dtype='float32',
                                           value=0.0)
            x_data = reviews[[*properties.training_features]].values
            return review_vectors.astype("float32"), x_data.astype("float32"), y_data.astype("float32")

    @staticmethod
    def clean_text(text: str, lowercase_text: bool = True, remove_punctuation: bool = True,
                   remove_stopwords: bool = True) -> List[str]:
        """
        Clean the text by lowercasing, removing punctuation, removing stopwords, and tokenizing.

        :param text: The input text to be cleaned.
        :param lowercase_text: Whether to convert the text to lowercase. Defaults to True.
        :param remove_punctuation: Whether to remove punctuation from the text. Defaults to True.
        :param remove_stopwords: Whether to remove stopwords from the text. Defaults to True.
        :return: The list of cleaned tokens.
        """
        if lowercase_text:
            text = text.lower()

        if remove_punctuation:
            translator = str.maketrans("", "", string.punctuation)
            text = text.translate(translator)

        tokens: List[str] = word_tokenize(text)

        if remove_stopwords:
            stop_words = set(stopwords.words("english"))
            tokens = [word for word in tokens if word not in stop_words]

        return tokens

    @staticmethod
    def lemmatize_words(words: List[str]) -> List[str]:
        """
        Lemmatize the given list of words.

        :param words: The list of words to be lemmatized.
        :return: The list of lemmatized words.
        """
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(word) for word in words]

    @staticmethod
    def stem_words(words: List[str]) -> List[str]:
        """
        Stem the given list of words.

        :param words: The list of words to be stemmed.
        :return: The list of stemmed words.
        """
        stemmer = PorterStemmer()
        return [stemmer.stem(word) for word in words]

    @staticmethod
    def get_tf_idf_vectorization(documents: List[str]) -> Any:
        """
        Get the TF-IDF vectorization of the given documents.

        :param documents: The list of documents to be vectorized.
        :return: The TF-IDF vectorization of the documents.
        """
        tfidf_vectorizer = TfidfVectorizer(lowercase=False, analyzer="word")
        return tfidf_vectorizer.fit_transform(documents)

    @staticmethod
    def clean_review_objects(reviews: pd.DataFrame, required_fields=None) -> pd.DataFrame:
        """
        Clean the review objects by dropping missing values, converting fields to appropriate data types,
        and adding a new field for review length.

        :param reviews: The DataFrame containing the review objects.
        :param required_fields: The list of required fields. Rows with null values for a required field will be dropped. Defaults to ["reviewText"].
        :return: The cleaned DataFrame of review objects.
        """
        if required_fields is None:
            required_fields = ["reviewText"]

        reviews = reviews.dropna(subset=required_fields, how="any")
        reviews["verified"] = reviews["verified"].astype(int)
        reviews.loc[:, "vote"] = (reviews["vote"]
                                  .str.replace(",", "")
                                  .fillna(0)
                                  .astype(int))
        reviews.loc[:, "reviewLength"] = reviews["reviewText"].str.len()

        reviews.loc[:, 'reviewTime'] = pd.to_datetime(reviews['reviewTime'], format='%m %d, %Y')
        most_recent_review = reviews['reviewTime'].max()
        reviews.loc[:, 'reviewAge'] = (most_recent_review - reviews['reviewTime']).apply(lambda x: x.days)

        return reviews

    @staticmethod
    def standardize_columns(reviews: pd.DataFrame, columns_to_standardize: List[str]) -> pd.DataFrame:
        """
        Normalizes 'vote' scores within each product group in a DataFrame.

        :param reviews: DataFrame with 'asin' for product IDs and 'vote' for scores.
        :return: Modified DataFrame with an added 'vote_std' column for normalized vote scores.
        """
        scaler = StandardScaler()

        for column in columns_to_standardize:
            reviews[f"{column}Std"] = scaler.fit_transform(reviews[[column]])

        return reviews
