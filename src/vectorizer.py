import os
from enum import Enum
from typing import List

import numpy as np
from gensim.models import Word2Vec, FastText
from keras.preprocessing.sequence import pad_sequences
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from get_logger import logger


class Vectorizer:
    """
    A class for vectorizing text.
    """

    class EmbeddingModel(Enum):
        WORD2VEC = 1
        FASTTEXT = 2

    word2vec_model_path = "../models/word2vec/trained_word2vec.model"
    word2vec_model = Word2Vec.load(word2vec_model_path) if os.path.isfile(word2vec_model_path) else None

    fasttext_model_path = "../models/fasttext/trained_fasttext.model"
    fasttext_model = FastText.load(fasttext_model_path) if os.path.isfile(fasttext_model_path) else None

    @staticmethod
    def get_tf_idf_vectors(documents: List[str]) -> csr_matrix:
        """
        Transforms a list of text documents into TF-IDF feature vectors.

        :param documents: A list of text documents to be vectorized.
        :return: The transformed TF-IDF feature vectors as a sparse matrix.
        """
        logger.info("Generating TF-IDF vectors.")
        tfidf_vectorizer = TfidfVectorizer(lowercase=False, analyzer="word")
        return tfidf_vectorizer.fit_transform(documents)

    @staticmethod
    def get_embeddings(documents: List[List[str]], model_type: EmbeddingModel, pad_vectors=True) -> np.ndarray:
        """
        Generates embeddings for a list of tokenized documents and optionally pads the sequences for uniform
        length.

        :param documents: A list of documents, each represented as a list of tokens (words).
        :param model_type: The type of embedding model to use
        :param pad_vectors: If True, pads the sequence of embeddings for each document to have uniform length.
                            Defaults to True.
        :return: A NumPy array containing the embeddings. If `pad_vectors` is False, the array will be of type
                 'object', with each element being a variable-length array of embeddings.

        """
        if model_type is Vectorizer.EmbeddingModel.WORD2VEC:
            model = Vectorizer.word2vec_model
        elif model_type is Vectorizer.EmbeddingModel.FASTTEXT:
            model = Vectorizer.fasttext_model
        else:
            raise ValueError("Unrecognized embedding model.")

        review_vectors = []
        for document in documents:
            review_vectors.append([model.wv[token] for token in document if token in model.wv])

        if pad_vectors:
            max_length = max(len(review) for review in review_vectors)
            return pad_sequences(review_vectors, maxlen=max_length, padding="post", dtype="float32",
                                 value=0.0)
        else:
            np.array(review_vectors, dtype=object)
