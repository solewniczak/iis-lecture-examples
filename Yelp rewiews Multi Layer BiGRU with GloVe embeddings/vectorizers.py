from collections import Counter, OrderedDict

import torch
from torchtext.data import get_tokenizer
from torchtext.vocab import vocab, GloVe

class ReviewVectorizer():
    def __init__(self, review_vocab, review_vectors, rating_vocab, max_size, max_review_words, tokenizer, language):
        self.review_vocab = review_vocab
        self.review_vectors = review_vectors
        self.rating_vocab = rating_vocab
        self.max_size = max_size
        self.max_review_words = max_review_words

        self.tokenizer = tokenizer
        self.language = language
        self.tokenizer_f = get_tokenizer(tokenizer, language)

    def _vectorize(self, indices):
        vector_length = (self.max_review_words, )

        pad_value = self.review_vocab['<pad>']
        vector = torch.full(vector_length, pad_value, dtype=torch.long)
        indeces_tensor = torch.tensor(indices, dtype=torch.long)
        vector[:len(indices)] = indeces_tensor

        return vector

    def vectorize(self, review_text):
        review_tokens = self.tokenizer_f(review_text)
        review_indices = [self.review_vocab[token] for token in review_tokens]
        review_indices = review_indices[:self.max_review_words]
        review_vector = self._vectorize(review_indices)

        return review_vector, len(review_indices)

    @classmethod
    def load_review_vocab_and_vectors_from_glove(cls):
        glove_vectors = GloVe(name='6B', dim=100)
        review_vocab = vocab(glove_vectors.stoi)
        review_vocab.append_token('<pad>')
        review_vocab.append_token('<unk>')
        review_vocab.set_default_index(review_vocab['<unk>'])

        review_vectors = torch.zeros([len(review_vocab), 100])
        review_vectors[:len(glove_vectors.vectors)] = glove_vectors.vectors

        return review_vocab, review_vectors

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

        review_vocab, review_vectors = cls.load_review_vocab_and_vectors_from_glove()

        rating_vocab = vocab(rating_counts)

        return cls(review_vocab, review_vectors, rating_vocab, max_size, max_review_words, tokenizer, language)

    @classmethod
    def from_serializable(cls, contents):
        rating_tokens = contents['rating_tokens']
        max_size = contents['max_size']
        max_review_words = contents['max_review_words']
        tokenizer = contents['tokenizer']
        language = contents['language']

        review_vocab, review_vectors = cls.load_review_vocab_and_vectors_from_glove()

        rating_vocab = vocab(OrderedDict([(token, 1) for token in rating_tokens]))

        return cls(review_vocab, review_vectors, rating_vocab, max_size, max_review_words, tokenizer, language)

    def to_serializable(self):
        return {'rating_tokens': self.rating_vocab.get_itos(),
                'max_size': self.max_size,
                'max_review_words': self.max_review_words,
                'tokenizer': self.tokenizer,
                'language': self.language}