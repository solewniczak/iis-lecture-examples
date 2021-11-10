from collections import Counter, OrderedDict

import torch
from torchtext.data import get_tokenizer
from torchtext.vocab import vocab


class ReviewVectorizer():
    def __init__(self, review_vocab, rating_vocab, max_size, max_review_words, tokenizer, language):
        self.review_vocab = review_vocab
        self.rating_vocab = rating_vocab
        self.max_size = max_size
        self.max_review_words = max_review_words

        self.tokenizer = tokenizer
        self.language = language

    def _vectorize(self, indices):
        vector_length = (self.max_review_words, )

        pad_value = self.review_vocab['<pad>']
        vector = torch.full(vector_length, pad_value, dtype=torch.long)
        indeces_tensor = torch.tensor(indices, dtype=torch.long)
        vector[:len(indices)] = indeces_tensor

        return vector

    def vectorize(self, review_text):
        tokenizer = get_tokenizer(self.tokenizer, self.language)
        review_tokens = tokenizer(review_text)
        review_indices = [self.review_vocab[token] for token in review_tokens]
        review_indices = review_indices[:self.max_review_words]
        review_vector = self._vectorize(review_indices)

        return review_vector, len(review_indices)


    @classmethod
    def from_reviews_list(cls, reviews, max_size, max_review_words, tokenizer='basic_english', language='en'):
        word_counts = Counter()
        rating_counts = Counter()
        tokenizer_f = get_tokenizer(tokenizer, language)
        for review in reviews:
            tokens = tokenizer_f(review['text'])
            for i, word in enumerate(tokens):
                if i == max_review_words:
                    break
                word_counts[word] += 1

            rating_counts[review['rating']] += 1

        review_vocab = vocab(OrderedDict(word_counts.most_common(max_size)))
        review_vocab.insert_token('<pad>', 0)
        review_vocab.insert_token('<unk>', 0)
        review_vocab.set_default_index(review_vocab['<unk>'])
        rating_vocab = vocab(rating_counts)

        return cls(review_vocab, rating_vocab, max_size, max_review_words, tokenizer, language)

    @classmethod
    def from_serializable(cls, contents):
        review_tokens = contents['review_tokens']
        rating_tokens = contents['rating_tokens']
        max_size = contents['max_size']
        max_review_words = contents['max_review_words']
        tokenizer = contents['tokenizer']
        language = contents['language']

        review_vocab = vocab(OrderedDict([(token, 1) for token in review_tokens]))
        review_vocab.set_default_index(review_vocab['<unk>'])

        rating_vocab = vocab(OrderedDict([(token, 1) for token in rating_tokens]))

        return cls(review_vocab, rating_vocab, max_size, max_review_words, tokenizer, language)

    def to_serializable(self):
        return {'review_tokens': self.review_vocab.get_itos(),
                'rating_tokens': self.rating_vocab.get_itos(),
                'max_size': self.max_size,
                'max_review_words': self.max_review_words,
                'tokenizer': self.tokenizer,
                'language': self.language}