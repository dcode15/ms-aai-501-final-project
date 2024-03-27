from typing import List, Any

import numpy as np
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer


class Vectorizer:
    """
    A class for vectorizing text.
    """

    @staticmethod
    def get_tf_idf_vectors(documents: List[str]) -> Any:
        """
        Transforms a list of text documents into TF-IDF feature vectors.

        :param documents: A list of text documents to be vectorized.
        :return: The transformed TF-IDF feature vectors as a sparse matrix. The exact type can vary depending
                 on the output of `TfidfVectorizer.fit_transform`.
        """
        tfidf_vectorizer = TfidfVectorizer(lowercase=False, analyzer="word")
        return tfidf_vectorizer.fit_transform(documents)

    @staticmethod
    def get_word2vec_embeddings(documents: List[List[str]], pad_vectors=True) -> np.ndarray:
        """
        Generates Word2Vec embeddings for a list of tokenized documents and optionally pads the sequences for uniform
        length.

        :param documents: A list of documents, each represented as a list of tokens (words).
        :param pad_vectors: If True, pads the sequence of embeddings for each document to have uniform length.
                            Defaults to True.
        :return: A NumPy array containing the embeddings. If `pad_vectors` is False, the array will be of type
                 'object', with each element being a variable-length array of embeddings.

        """
        w2v_model = Word2Vec(sentences=documents, vector_size=100, window=5, min_count=1, workers=4)
        review_vectors = []
        for document in documents:
            review_vectors.append([w2v_model.wv[token] for token in document if token in w2v_model.wv])

        if pad_vectors:
            max_length = max(len(review) for review in review_vectors)
            return pad_sequences(review_vectors, maxlen=max_length, padding='post', dtype='float32',
                                 value=0.0)
        else:
            np.array(review_vectors, dtype=object)
