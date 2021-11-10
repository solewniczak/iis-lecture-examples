import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)


def attention(q, k, v, d_k, mask):
    k = k.transpose(-2, -1) # transpose to: batch_size x heads x d_k x seq_len
    scores = torch.matmul(q, k) / math.sqrt(d_k) # batch_size x heads x seq_len x seq_len

    mask = mask.unsqueeze(1)  # brodcast to apply for every head
    scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)

    output = torch.matmul(scores, v)
    return output # batch_size x heads x seq_len x d_k


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)  # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.out(concat)
        return output

input_vocabulary = ['<sos>', '<eos>', 'I', "'m", 'not', 'letting', 'Tom', 'go']
output_vocabulary = ['<sos>', '<eos>', 'Nie', 'pozwolę', 'Tomowi', 'iść']

encoder_input = torch.LongTensor([[0,2,3,4,5,6,7,1]]).transpose(1,0) # seq_len x batch
decoder_input = torch.LongTensor([[0,2,3,4,5]]).transpose(1,0) # seq_len x batch
decoder_target = torch.LongTensor([[2,3,4,5,1]]).transpose(1,0) # seq_len x batch

batch_size = 1
embeding_dim = 4
encoder_embed = nn.Embedding(num_embeddings=8, embedding_dim=embeding_dim)
decoder_embed = nn.Embedding(num_embeddings=6, embedding_dim=embeding_dim)

encoder_embeddings = encoder_embed(encoder_input).transpose(1, 0) # batch_size x seq_len x encoder_embed_size
decoder_input_embeddings = decoder_embed(decoder_input).transpose(1, 0) # batch_size x seq_len x encoder_embed_size
decoder_target_embeddings = decoder_embed(decoder_input).transpose(1, 0) # batch_size x seq_len x encoder_embed_size

heads = 2
d_model = embeding_dim

encoder_self_attention = MultiHeadAttention(heads, d_model)
masked_decoder_self_attention = MultiHeadAttention(heads, d_model)
enocder_decoder_attention = MultiHeadAttention(heads, d_model)

source_mask = torch.ones((batch_size, encoder_embeddings.shape[1]))
encoder_outputs = encoder_self_attention(encoder_embeddings, encoder_embeddings, encoder_embeddings, source_mask)

# Decoding time step = 0
size = decoder_input_embeddings.shape[1]
decoder_input_mask = torch.triu(torch.ones((batch_size, size, size), dtype=torch.uint8), diagonal=1)
decoder_input_mask = (decoder_input_mask == 0)
decoder_input_outputs = masked_decoder_self_attention(decoder_input_embeddings, decoder_input_embeddings, decoder_input_embeddings, decoder_input_mask)
enocder_decoder_attention = enocder_decoder_attention(decoder_input_outputs, encoder_outputs, encoder_outputs, source_mask)