import string
from typing import List

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import StandardScaler

from enums import TextNormalizationStrategy, TokenizationStrategy
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
    def preprocess_reviews(reviews: pd.DataFrame, lowercase_text=True, remove_punctuation=True,
                           remove_stopwords=True, tokenization_strategy=TokenizationStrategy.NONE,
                           text_normalization_strategy=TextNormalizationStrategy.NONE) -> pd.DataFrame:
        """
        Preprocesses a DataFrame of reviews, performing various cleaning and normalization steps.

        The function performs the following operations:
        1. Cleans the review objects by dropping missing values, converting fields to appropriate data types,
        and adding a new reviewAge and reviewLength fields.
        2. Standardizes certain columns (e.g., 'vote', 'reviewLength', 'reviewAge') for uniformity.
        3. Applies text cleaning to the 'reviewText' column, including optional lowercase conversion, punctuation removal,
           and stopwords removal based on the parameters provided.
        4. Depending on the `text_normalization_strategy`, it may apply stemming or lemmatization to the text
           to reduce words to their base or dictionary form.
        5. If `tokenization_strategy` is not NONE, the cleaned review texts are left tokenized; otherwise, they are
           joined back into strings.

        :param reviews: A DataFrame containing the reviews to preprocess. Must include a 'reviewText' column.
        :param lowercase_text: If True, converts all characters in the review text to lowercase. Defaults to True.
        :param remove_punctuation: If True, removes all punctuation characters from the review text. Defaults to True.
        :param remove_stopwords: If True, removes common stopwords from the review text. Defaults to True.
        :param tokenization_strategy: Specifies the strategy for tokenizing the text. The default
                                      value, TokenizationStrategy.NONE, means no tokenization is applied post-cleaning.
        :param text_normalization_strategy: Specifies the strategy for normalizing the text,
                                            either through stemming or lemmatization. The default value,
                                            TextNormalizationStrategy.NONE, applies no normalization.
        :return: The input DataFrame with an additional 'cleanedReviewText' column containing the preprocessed text.
        """
        logger.info("Preprocessing data.")
        reviews = Preprocessor.clean_review_objects(reviews)
        reviews = Preprocessor.standardize_columns(reviews, ["vote", "reviewLength", "reviewAge"])
        reviews["cleanedReviewText"] = reviews["reviewText"].apply(Preprocessor.clean_text,
                                                                   lowercase_text=lowercase_text,
                                                                   remove_punctuation=remove_punctuation,
                                                                   remove_stopwords=remove_stopwords)

        if text_normalization_strategy is TextNormalizationStrategy.STEMMING:
            reviews["cleanedReviewText"] = reviews["cleanedReviewText"].apply(Preprocessor.stem_words)
        elif text_normalization_strategy is TextNormalizationStrategy.LEMMATIZATION:
            reviews["cleanedReviewText"] = reviews["cleanedReviewText"].apply(Preprocessor.lemmatize_words)

        if tokenization_strategy is TokenizationStrategy.NONE:
            reviews["cleanedReviewText"] = reviews["cleanedReviewText"].apply(lambda tokens: " ".join(tokens))

        return reviews

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
    def clean_review_objects(reviews: pd.DataFrame, required_fields=None) -> pd.DataFrame:
        """
        Cleans a DataFrame of review objects by performing various data cleaning operations,
        including handling missing values, type conversions, and generating new features.

        :param reviews: The DataFrame containing review data that needs to be cleaned.
        :param required_fields: A list of column names considered essential for a review to be valid.
                                Defaults to ["reviewText"] if not provided.
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

        reviews.loc[:, "reviewTime"] = pd.to_datetime(reviews["reviewTime"], format="%m %d, %Y")
        most_recent_review = reviews["reviewTime"].max()
        reviews.loc[:, "reviewAge"] = (most_recent_review - reviews["reviewTime"]).apply(lambda x: x.days)

        return reviews

    @staticmethod
    def standardize_columns(reviews: pd.DataFrame, columns_to_standardize: List[str]) -> pd.DataFrame:
        """
        Standardizes specified columns in the given DataFrame by applying z-score normalization to the specified columns.
        This transforms the data in the column to have a mean of 0 and a standard deviation of 1.

        :param reviews: The DataFrame containing the data to be standardized.
        :param columns_to_standardize: A list of column names in the DataFrame that should be standardized.
        :return: The original DataFrame with additional columns for each standardized column. The names of these
                 new columns are the original column names suffixed with 'Std'.
        """

        scaler = StandardScaler()

        for column in columns_to_standardize:
            reviews[f"{column}Std"] = scaler.fit_transform(reviews[[column]])

        return reviews
