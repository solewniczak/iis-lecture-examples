import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from args import args
from torch.nn import functional as F


class NMTEncoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, padding_idx):

        super().__init__()
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=padding_idx)
        self.rnn = nn.GRU(embedding_size, hidden_size, bidirectional=True, batch_first=True)

    def forward(self, x_in, x_lengths):
        x_embedded = self.embedding(x_in)

        # create PackedSequence; x_packed.data.shape=(number_items, embeddign_size)
        x_packed = pack_padded_sequence(x_embedded, x_lengths.cpu(), batch_first=True)
        x_birnn_out, x_birnn_h = self.rnn(x_packed)
        # permute to (batch_size, num_rnn, feature_size)
        x_birnn_h = x_birnn_h.permute(1, 0, 2)

        # flatten features; reshape to (batch_size, num_rnn * feature_size)
        x_birnn_h = x_birnn_h.contiguous().view(x_birnn_h.size(0), -1)

        return x_birnn_h


class NMTDecoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, padding_idx, max_target_size, sos_index):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_target_size = max_target_size + 1 # +1 for EOS token
        self.sos_index = sos_index

        self.embedding = nn.Embedding(output_size, embedding_size, padding_idx=padding_idx)
        self.gru_cell = nn.GRUCell(embedding_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, output_size)

    def _init_indices(self, size):
        """ return the START-OF-SEQUENCE index vector """
        return torch.ones(size, dtype=torch.long, device=args.device) * self.sos_index

    def forward(self, initial_hidden_state, target_sequence=None, sample_probability=0.0):

        batch_size = initial_hidden_state.size(0)

        if target_sequence is None:
            sample_probability = 1.0
            output_sequence_size = self.max_target_size
        else:
            # We are making an assumption there: The batch is on first
            # The input is (Batch, Seq)
            # We want to iterate over sequence so we permute it to (S, B)
            target_sequence = target_sequence.permute(1, 0)
            output_sequence_size = target_sequence.size(0)

        # initialize first y_t word as SOS
        y_t_index = self._init_indices(batch_size)
        h_t = initial_hidden_state
        output_vectors = []
        use_sample = torch.rand(1).item() < sample_probability
        for i in range(output_sequence_size):
            if not use_sample:
                y_t_index = target_sequence[i]

            # Step 1: Embed word
            y_input_vector = self.embedding(y_t_index)

            # Step 2: Make a GRU step, getting a new hidden vector
            h_t = self.gru_cell(y_input_vector, h_t)

            # Step 3: Use the current hidden and context vectors to make a prediction to the next word
            score_for_y_t_index = self.classifier(h_t)

            if use_sample:
                # p_y_t_index = F.softmax(score_for_y_t_index, dim=1)
                _, y_t_index = torch.max(score_for_y_t_index, dim=1)

            # collect the prediction scores
            output_vectors.append(score_for_y_t_index)

        output_vectors = torch.stack(output_vectors).permute(1, 0, 2)

        return output_vectors


class NMTModel(nn.Module):
    def __init__(self, source_vocab_size, source_embedding_size, source_padding_idx,
                 target_vocab_size, target_embedding_size, target_padding_idx, encoding_size,
                 max_target_size, sos_index):
        super().__init__()
        self.encoder = NMTEncoder(input_size=source_vocab_size,
                                  embedding_size=source_embedding_size,
                                  hidden_size=encoding_size,
                                  padding_idx=source_padding_idx)

        decoding_size = 2*encoding_size
        self.decoder = NMTDecoder(output_size=target_vocab_size,
                                  embedding_size=target_embedding_size,
                                  hidden_size=decoding_size,
                                  padding_idx=target_padding_idx,
                                  max_target_size=max_target_size,
                                  sos_index=sos_index)

    def forward(self, x_source, x_source_lengths, target_sequence=None, sample_probability=0.0):
        final_hidden_states = self.encoder(x_source, x_source_lengths)
        decoded_states = self.decoder(initial_hidden_state=final_hidden_states,
                                      target_sequence=target_sequence,
                                      sample_probability=sample_probability)
        return decoded_states