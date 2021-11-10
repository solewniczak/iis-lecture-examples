from collections import Counter

import torch

from helper import preprocess_text
from vocabulary import Vocabulary


class ReviewVectorizer(object):
    def __init__(self, review_vocab, rating_vocab, words_limit=1000, max_review_words=100):
        self.review_vocab = review_vocab
        self.rating_vocab = rating_vocab
        self._words_limit = words_limit
        self._max_review_words = max_review_words

    def vectorize(self, review_text):
        review_text = preprocess_text(review_text)

        one_hot_matrix_size = (len(self.review_vocab), self._max_review_words)
        one_hot_matrix = torch.zeros(one_hot_matrix_size, dtype=torch.float)

        review_words = review_text.split()
        for position_index, token in enumerate(review_words[:self._max_review_words]):
            word_index = self.review_vocab.lookup_token(token)
            one_hot_matrix[word_index][position_index] = 1

        return one_hot_matrix

    @classmethod
    def from_json(cls, reviews, words_limit=1000, max_review_words=100):
        review_vocab = Vocabulary(add_unk=True)
        rating_vocab = Vocabulary(add_unk=False)

        rating_vocab.add_many(['positive', 'neutral', 'negative'])

        # Add top words if count > provided count
        word_counts = Counter()
        for review in reviews:
            review_text = preprocess_text(review['text'])
            for word in review_text.split():
                word_counts[word] += 1

        for word, count in word_counts.most_common(words_limit):
            review_vocab.add_token(word)

        return cls(review_vocab, rating_vocab, words_limit, max_review_words)

    @classmethod
    def from_serializable(cls, contents):
        review_vocab = Vocabulary.from_serializable(contents['review_vocab'])
        rating_vocab = Vocabulary.from_serializable(contents['rating_vocab'])
        return cls(review_vocab=review_vocab, rating_vocab=rating_vocab,
                   words_limit=contents['words_limit'], max_review_words=contents['max_review_words'])

    def to_serializable(self):
        return {'review_vocab': self.review_vocab.to_serializable(),
                'rating_vocab': self.rating_vocab.to_serializable(),
                'words_limit': self._words_limit,
                'max_review_words': self._max_review_words}