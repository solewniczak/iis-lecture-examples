import math
import torch
import torch.nn as nn

torch.manual_seed(1337)

input_vocabulary = ['<sos>', '<eos>', 'I', "'m", 'not', 'letting', 'Tom', 'go']
encoder_input = torch.LongTensor([[0,2,3,4,5,6,7,1]]).transpose(1,0) # seq_len x batch

batch_size = 1
embeding_dim = 4
encoder_embed = nn.Embedding(num_embeddings=8, embedding_dim=embeding_dim)

def positional_encoding(embeddings):
    seq_len, batch_size, d_model = embeddings.size()
    pe = torch.zeros(seq_len, d_model)
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
            pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

    pe = pe.unsqueeze(0) # batch x seq_len x embedding_dim
    return pe

encoder_embeddings = encoder_embed(encoder_input) # seq_len x batch_size x encoder_embed_size
pe = positional_encoding(encoder_embeddings)
encoder_embeddings = encoder_embeddings +  pe.transpose(0,1)