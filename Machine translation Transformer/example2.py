import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

input_vocabulary = ['<sos>', '<eos>', 'I', "'m", 'not', 'letting', 'Tom', 'go']
encoder_input = torch.LongTensor([[0,2,3,4,5,6,7,1]]).transpose(1,0) # seq_len x batch

batch_size = 1
embeding_dim = 4
encoder_embed = nn.Embedding(num_embeddings=8, embedding_dim=embeding_dim)


def attention(q, k, v, d_k, mask):
    k = k.transpose(-2, -1) # transpose to: batch_size x heads x d_k x seq_len
    scores = torch.matmul(q, k) / math.sqrt(d_k) # batch_size x heads x seq_len x seq_len

    mask = mask.unsqueeze(1)
    scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)

    output = torch.matmul(scores, v)
    return output # batch_size x heads x seq_len x d_k

encoder_embeddings = encoder_embed(encoder_input).transpose(1, 0) # batch_size x seq_len x encoder_embed_size

heads = 2
d_model = embeding_dim
d_k = d_model // heads

q_linear = nn.Linear(d_model, d_model)
v_linear = nn.Linear(d_model, d_model)
k_linear = nn.Linear(d_model, d_model)
out = nn.Linear(d_model, d_model)

# perform linear operation and split into h heads
k = k_linear(encoder_embeddings).view(batch_size, -1, heads, d_k)
q = q_linear(encoder_embeddings).view(batch_size, -1, heads, d_k)
v = v_linear(encoder_embeddings).view(batch_size, -1, heads, d_k)

# transpose to get dimensions batch_size x heads x seq_len x d_k
k = k.transpose(1, 2)
q = q.transpose(1, 2)
v = v.transpose(1, 2)  # calculate attention
mask = torch.ones((batch_size, encoder_embeddings.shape[1]))
scores = attention(q, k, v, d_k, mask)

# concatenate heads and put through final linear layer
concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, d_model)

output = out(concat)