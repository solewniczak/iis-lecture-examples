import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class ReviewClassifierBiGRU(nn.Module):
    def __init__(self, embeddings, hidden_dim, output_dim, padding_idx, num_layers, dropout):

        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=True, padding_idx=padding_idx)
        self.rnn = nn.GRU(embeddings.shape[1], hidden_dim,
                          batch_first=True,
                          num_layers=num_layers,
                          bidirectional=True,
                          dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_in, x_lengths):
        x_embedded = self.embedding(x_in)

        # create PackedSequence; x_packed.data.shape=(number_items, embeddign_size)
        x_packed = pack_padded_sequence(x_embedded, x_lengths.cpu(), batch_first=True)

        output, hidden = self.rnn(x_packed)

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        return self.fc(hidden.squeeze(0))

