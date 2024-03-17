import string
from typing import List, Any

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


class PreProcessor:
    """
    A class for preprocessing text data.
    """
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("punkt")

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
        reviews.loc[:, 'verified'] = reviews['verified'].astype(int)
        reviews.loc[:, "vote"] = (reviews["vote"]
                                  .str.replace(",", "")
                                  .fillna(0)
                                  .astype(int))
        reviews.loc[:, "reviewLength"] = reviews["reviewText"].str.len()

        return reviews
