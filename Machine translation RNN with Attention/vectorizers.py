from collections import Counter

import torch
from torchtext.vocab import Vocab


class NMTVectorizer():
    UNK = '<unk>'
    PAD = '<pad>'
    SOS = '<sos>'
    EOS = '<eos>'
    SPECIALS = (UNK, PAD, SOS, EOS)

    def __init__(self, source_counter, target_counter, max_words):
        self.source_vocab = Vocab(source_counter, specials=self.SPECIALS)
        self.target_vocab = Vocab(target_counter, specials=self.SPECIALS)
        self.max_words = max_words

    def _vectorize(self, indices, vector_length=-1):
        if vector_length < 0:
            vector_length = len(indices)

        vector_length = (vector_length,)

        pad_value = self.source_vocab[self.PAD]
        vector = torch.full(vector_length, pad_value, dtype=torch.long)
        indeces_tensor = torch.tensor(indices, dtype=torch.long)
        vector[:len(indices)] = indeces_tensor

        return vector

    def _get_source_indices(self, text):
        indices = [self.source_vocab[self.SOS]]
        indices.extend(self.source_vocab[token] for token in text.split())
        indices.append(self.source_vocab[self.EOS])
        return indices

    def _get_target_indices(self, text):
        indices = [self.target_vocab[token] for token in text.split()]
        x_indices = [self.target_vocab[self.SOS]] + indices
        y_indices = indices + [self.target_vocab[self.EOS]]
        return x_indices, y_indices

    def vectorize(self, source_text, target_text):
        source_vector_length = self.max_words + 2
        target_vector_length = self.max_words + 1

        source_indices = self._get_source_indices(source_text)
        source_vector = self._vectorize(source_indices, source_vector_length)

        target_x_indices, target_y_indices = self._get_target_indices(target_text)
        target_x_vector = self._vectorize(target_x_indices, target_vector_length)
        target_y_vector = self._vectorize(target_y_indices, target_vector_length)

        return {"source_vector": source_vector,
                "target_x_vector": target_x_vector,
                "target_y_vector": target_y_vector,
                "source_length": len(source_indices)}

    def vectorize_source(self, source_text):
        source_vector_length = self.max_words + 2
        source_indices = self._get_source_indices(source_text)
        source_vector = self._vectorize(source_indices, source_vector_length)

        return source_vector, source_vector_length

    @classmethod
    def from_pairs(cls, pairs, max_words):
        source_counter = Counter()
        target_counter = Counter()
        for pair in pairs:
            for word in pair[0].split():
                source_counter[word] += 1
            for word in pair[1].split():
                target_counter[word] += 1

        return cls(source_counter, target_counter, max_words)

    @classmethod
    def from_serializable(cls, contents):
        source_counter = Counter(contents['source_counter'])
        target_counter = Counter(contents['target_counter'])
        max_words = contents['max_words']

        return cls(source_counter, target_counter, max_words)

    def to_serializable(self):
        return {'source_counter': self.source_vocab.freqs,
                'target_counter': self.target_vocab.freqs,
                'max_words': self.max_words}