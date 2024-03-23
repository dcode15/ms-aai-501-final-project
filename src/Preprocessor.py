import string
from typing import List, Any, Tuple

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from PropertyEnums import TextNormalizationStrategy, VectorizationStrategy
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
        import properties_template as properties
        # properties = importlib.import_module(properties_file)

        logger.info("Loading data.")
        reviews = pd.read_json(properties.data_file_path, lines=True)
        logger.info("Cleaning data.")
        reviews = Preprocessor.clean_review_objects(reviews)
        reviews = Preprocessor.normalize_votes(reviews)
        reviews["cleanedTokens"] = reviews["reviewText"].apply(Preprocessor.clean_text,
                                                               lowercase_text=properties.lowercase_text,
                                                               remove_punctuation=properties.remove_punctuation,
                                                               remove_stopwords=properties.remove_stopwords)

        if properties.text_normalization_strategy == TextNormalizationStrategy.STEMMING:
            reviews["cleanedTokens"] = reviews["cleanedTokens"].apply(Preprocessor.stem_words)
        elif properties.text_normalization_strategy == TextNormalizationStrategy.LEMMATIZATION:
            reviews["cleanedTokens"] = reviews["cleanedTokens"].apply(Preprocessor.lemmatize_words)

        logger.info("Vectorizing reviews.")
        if properties.vectorization_strategy == VectorizationStrategy.TF_IDF:
            reviews["cleanedReview"] = reviews["cleanedTokens"].apply(lambda tokens: " ".join(tokens))
            review_vectors = Preprocessor.get_tf_idf_vectorization(reviews["cleanedReview"])

        hstack_args = [reviews[feature].values[:, None] for feature in properties.training_features]
        x_data = hstack([review_vectors, *hstack_args])
        y_data = reviews["vote_std"]

        return x_data, y_data

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
        reviews.loc[:, "verified"] = reviews["verified"].astype(int)
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
    def normalize_votes(reviews: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes 'vote' scores within each product group in a DataFrame.

        :param reviews: DataFrame with 'asin' for product IDs and 'vote' for scores.
        :return: Modified DataFrame with an added 'vote_std' column for normalized vote scores.
        """
        std_devs = reviews.groupby("asin")["vote"].transform("std")
        df_filtered = reviews[std_devs > 0].copy()
        df_filtered.loc[:, "vote_std"] = df_filtered.groupby("asin")["vote"].transform(
            lambda x: (x - x.mean()) / x.std())
        return df_filtered
