from enum import Enum


class TokenizationType(Enum):
    WORD = 1


class TextNormalizationStrategy(Enum):
    NONE = 1
    STEMMING = 2
    LEMMATIZATION = 3


class VectorizationStrategy(Enum):
    TF_IDF = 1
