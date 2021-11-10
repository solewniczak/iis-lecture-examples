import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
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

        x_unpacked, _ = pad_packed_sequence(x_birnn_out, batch_first=True)

        return x_unpacked, x_birnn_h


def attention(encoder_state_vectors, query_vector):
    """
    Args:
        encoder_state_vectors (torch.Tensor): 3dim tensor from bi-GRU in encoder
        query_vector (torch.Tensor): hidden state in decoder GRU
    """
    batch_size, num_vectors, vector_size = encoder_state_vectors.size()
    vector_scores = torch.sum(encoder_state_vectors * query_vector.view(batch_size, 1, vector_size), dim=2)
    vector_probabilities = F.softmax(vector_scores, dim=1)
    weighted_vectors = encoder_state_vectors * vector_probabilities.view(batch_size, num_vectors, 1)
    context_vectors = torch.sum(weighted_vectors, dim=1)
    return context_vectors, vector_probabilities, vector_scores

class NMTDecoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, padding_idx, max_target_size, sos_index):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_target_size = max_target_size + 1 # +1 for EOS token
        self.sos_index = sos_index

        self.embedding = nn.Embedding(output_size, embedding_size, padding_idx=padding_idx)
        self.gru_cell = nn.GRUCell(embedding_size + hidden_size, hidden_size) #  NEW: + hidden_size for input vector
        self.classifier = nn.Linear(hidden_size * 2, output_size) # NEW: hidden_size * 2 for attention

    def _init_indices(self, batch_size, device):
        """ return the START-OF-SEQUENCE index vector """
        return torch.ones(batch_size, dtype=torch.long, device=device) * self.sos_index

    def _init_context_vectors(self, batch_size, device):
        """ return a zeros vector for initializing the context """
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def forward(self, encoder_state, initial_hidden_state, target_sequence=None, sample_probability=0.0): # encoder_state <- NEW
        if target_sequence is None:
            sample_probability = 1.0
            output_sequence_size = self.max_target_size
        else:
            # We are making an assumption there: The batch is on first
            # The input is (Batch, Seq)
            # We want to iterate over sequence so we permute it to (S, B)
            target_sequence = target_sequence.permute(1, 0)
            output_sequence_size = target_sequence.size(0)

        batch_size = encoder_state.size(0)
        # initialize context vectors to zeros
        context_vectors = self._init_context_vectors(batch_size, encoder_state.device)
        # initialize first y_t word as SOS
        y_t_index = self._init_indices(batch_size, encoder_state.device)
        h_t = initial_hidden_state

        output_vectors = []
        attention_matrixes = []

        use_sample = torch.rand(1).item() < sample_probability
        for i in range(output_sequence_size):
            if not use_sample:
                y_t_index = target_sequence[i]

            # Step 1: Embed word (and concat with previous context)
            y_input_vector = self.embedding(y_t_index)
            rnn_input = torch.cat([y_input_vector, context_vectors], dim=1)

            # Step 2: Make a GRU step, getting a new hidden vector
            h_t = self.gru_cell(rnn_input, h_t)

            # Step 3: Use the current hidden to attend to the encoder state
            context_vectors, p_attn, _ = attention(encoder_state_vectors=encoder_state, query_vector=h_t)
            attention_matrixes.append(p_attn.detach())

            # Step 4: Use the current hidden and context vectors to make a prediction to the next word
            prediction_vector = torch.cat((context_vectors, h_t), dim=1)
            score_for_y_t_index = self.classifier(prediction_vector)

            if use_sample:
                # p_y_t_index = F.softmax(score_for_y_t_index, dim=1)
                _, y_t_index = torch.max(score_for_y_t_index, dim=1)

            # collect the prediction scores
            output_vectors.append(score_for_y_t_index)

        output_vectors = torch.stack(output_vectors).permute(1, 0, 2) # batch, seq
        attention_matrixes = torch.stack(attention_matrixes).permute(1, 0, 2) # batch, seq

        return output_vectors, attention_matrixes


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
        encoder_state, final_hidden_states = self.encoder(x_source, x_source_lengths)
        decoded_states, attention_matrixes = self.decoder(encoder_state=encoder_state,
                                      initial_hidden_state=final_hidden_states,
                                      target_sequence=target_sequence,
                                      sample_probability=sample_probability)
        return decoded_states, attention_matrixes