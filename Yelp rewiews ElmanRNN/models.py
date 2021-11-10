import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class ReviewClassifierRNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, padding_idx):

        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=padding_idx)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_in, x_lengths):
        x_embedded = self.embedding(x_in)

        # create PackedSequence; x_packed.data.shape=(number_items, embeddign_size)
        x_packed = pack_padded_sequence(x_embedded, x_lengths, batch_first=True)

        output, hidden = self.rnn(x_packed)

        return self.fc(hidden.squeeze(0))