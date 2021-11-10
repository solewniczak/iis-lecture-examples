import json

from torch.utils.data import Dataset, DataLoader

from vectorizers import NMTVectorizer


class NMTDataset(Dataset):
    def __init__(self, pairs, vectorizer):
        self._pairs = pairs
        self._vectorizer = vectorizer

    @staticmethod
    def read_pairs(dataset_filepath):
        pairs = []
        with open(dataset_filepath, 'r') as fp:
            for line in fp.readlines():
                cols = line.strip().split('\t')
                pairs.append((cols[0], cols[1]))
        return pairs

    @classmethod
    def load_dataset_and_make_vectorizer(cls, dataset_filepath, max_words):
        pairs = cls.read_pairs(dataset_filepath)
        vectorizer = NMTVectorizer.from_pairs(pairs, max_words)
        return cls(pairs, vectorizer)

    @classmethod
    def load_dataset_and_load_vectorizer(cls, dataset_filepath, vectorizer_filepath):
        pairs = cls.read_pairs(dataset_filepath)
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)
        return cls(pairs, vectorizer)

    @staticmethod
    def load_vectorizer_only(vectorizer_filepath):
        with open(vectorizer_filepath) as fp:
            return NMTVectorizer.from_serializable(json.load(fp))

    def save_vectorizer(self, vectorizer_filepath):
        with open(vectorizer_filepath, 'w') as fp:
            json.dump(self._vectorizer.to_serializable(), fp)

    def get_vectorizer(self):
        return self._vectorizer

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, index):
        pair = self._pairs[index]
        vector_dict = self._vectorizer.vectorize(pair[0], pair[1])

        return {"x_source": vector_dict["source_vector"],
                "x_target": vector_dict["target_x_vector"],
                "y_target": vector_dict["target_y_vector"],
                "x_source_length": vector_dict["source_length"]}


def get_num_batches(dataset, batch_size):
    return (len(dataset) + batch_size - 1) // batch_size


def generate_rnn_batches(dataset, batch_size, shuffle=True, drop_last=True, device="cpu"):
    """A generator function which wraps the PyTorch DataLoader.  The NMT Version """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        lengths = data_dict['x_source_length']
        sorted_length_indices = lengths.argsort(descending=True).tolist()

        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name][sorted_length_indices].to(device)
        yield out_data_dict