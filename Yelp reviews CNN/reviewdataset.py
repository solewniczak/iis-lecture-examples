import json

from torch.utils.data import Dataset, DataLoader

from reviewvectorizer import ReviewVectorizer


class ReviewDataset(Dataset):
    def __init__(self, reviews, vectorizer):
        self._reviews = reviews
        self._vectorizer = vectorizer

    @staticmethod
    def read_json(reviews_json_filepath, limit=-1):
        if type(limit) is int:
           limit = (limit,)
        if len(limit) == 1:
            limit = (0, limit[0])

        start, end = limit
        reviews = []
        with open(reviews_json_filepath) as fp:
            for i, line in enumerate(fp):
                if i < start:
                    continue
                if end != -1 and i >= end:
                    break
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
    def load_dataset_and_make_vectorizer(cls, reviews_json_filepath, words_limit=1000, reviews_limit=(0, 5000), max_review_words=100):
        reviews = cls.read_json(reviews_json_filepath, limit=reviews_limit)
        vectorizer = ReviewVectorizer.from_json(reviews, words_limit=words_limit, max_review_words=max_review_words)
        return cls(reviews, vectorizer)

    @classmethod
    def load_dataset_and_load_vectorizer(cls, reviews_json_filepath, vectorizer_filepath, reviews_limit=(0, 5000)):
        reviews = cls.read_json(reviews_json_filepath, limit=reviews_limit)
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

        review_vector = self._vectorizer.vectorize(row['text'])
        rating_index = self._vectorizer.rating_vocab.lookup_token(row['rating'])

        return {'x_data': review_vector, 'y_target': rating_index}


def get_num_batches(dataset, batch_size):
    return (len(dataset) + batch_size - 1) // batch_size


def generate_batches(dataset, batch_size, shuffle=True, drop_last=True, device="cpu"):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict