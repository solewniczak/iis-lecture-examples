import json

from torch.utils.data import Dataset, DataLoader

from vectorizers import ReviewVectorizer


class ReviewDataset(Dataset):
    def __init__(self, reviews, vectorizer):
        self._reviews = reviews
        self._vectorizer = vectorizer

    @staticmethod
    def read_json(reviews_json_filepath):
        reviews = []
        with open(reviews_json_filepath) as fp:
            for i, line in enumerate(fp):
                line = line.strip()
                json_line = json.loads(line)

                if json_line['stars'] >= 4.0:
                    rating = 'positive'
                elif json_line['stars'] <= 2.0:
                    rating = 'negative'
                else:
                    rating = 'neutral'

                reviews.append({
                    'text': json_line['text'],
                    'rating': rating,
                })
        return reviews

    @classmethod
    def load_dataset_and_make_vectorizer(cls, reviews_json_filepath, max_size, max_rewivew_words, tokenizer='basic_english', language='en'):
        reviews = cls.read_json(reviews_json_filepath)
        vectorizer = ReviewVectorizer.from_reviews_list(reviews, max_size, max_rewivew_words, tokenizer, language)
        return cls(reviews, vectorizer)

    @classmethod
    def load_dataset_and_load_vectorizer(cls, reviews_json_filepath, vectorizer_filepath):
        reviews = cls.read_json(reviews_json_filepath)
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)
        return cls(reviews, vectorizer)

    @staticmethod
    def load_vectorizer_only(vectorizer_filepath):
        with open(vectorizer_filepath) as fp:
            return ReviewVectorizer.from_serializable(json.load(fp))

    def save_vectorizer(self, vectorizer_filepath):
        with open(vectorizer_filepath, 'w') as fp:
            json.dump(self._vectorizer.to_serializable(), fp)

    def get_vectorizer(self):
        return self._vectorizer

    def __len__(self):
        return len(self._reviews)

    def __getitem__(self, index):
        row = self._reviews[index]

        review_vectors, review_lengths = self._vectorizer.vectorize(row['text'])
        rating_index = self._vectorizer.rating_vocab[row['rating']]

        return {'x_data': review_vectors, 'x_length': review_lengths, 'y_target': rating_index}


def get_num_batches(dataset, batch_size):
    return (len(dataset) + batch_size - 1) // batch_size


def generate_rnn_batches(dataset, batch_size, shuffle=True,
                         drop_last=True, device="cpu"):
    """A generator function which wraps the PyTorch DataLoader.  The NMT Version """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        lengths = data_dict['x_length']
        sorted_length_indices = lengths.argsort(descending=True).tolist()

        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name][sorted_length_indices].to(device)
        yield out_data_dict