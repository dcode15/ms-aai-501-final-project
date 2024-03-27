from enum import Enum


class TokenizationStrategy(Enum):
    WORD = 1
    NONE = 2


class TextNormalizationStrategy(Enum):
    NONE = 1
    STEMMING = 2
    LEMMATIZATION = 3


class VectorizationStrategy(Enum):
    TF_IDF = 1
    WORD_2_VEC = 2
