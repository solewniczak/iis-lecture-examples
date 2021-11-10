import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

input_vocabulary = ['<sos>', '<eos>', 'I', "'m", 'not', 'letting', 'Tom', 'go']
output_vocabulary = ['<sos>', '<eos>', 'Nie', 'pozwolę', 'Tomowi', 'iść']

encoder_input = torch.LongTensor([[0,2,3,4,5,6,7,1]]).transpose(1,0) # seq_len x batch
decoder_input = torch.LongTensor([[0,2,3,4,5]]).transpose(1,0) # seq_len x batch
decoder_target = torch.LongTensor([[2,3,4,5,1]]).transpose(1,0) # seq_len x batch

batch_size = 1
embeding_dim = 3
encoder_hidden_size = 2
decoder_hidden_size = encoder_hidden_size

encoder_embed = nn.Embedding(num_embeddings=8, embedding_dim=embeding_dim)
decoder_embed = nn.Embedding(num_embeddings=6, embedding_dim=embeding_dim)
encoder = nn.RNN(input_size=embeding_dim, hidden_size=encoder_hidden_size, bias=False, nonlinearity='relu')
decoder_cell = nn.RNNCell(input_size=embeding_dim+decoder_hidden_size, hidden_size=decoder_hidden_size, bias=False, nonlinearity='relu')

encoder_embeddings = encoder_embed(encoder_input) # seq_len x batch_size x encoder_embed_size
decoder_input_embeddings = decoder_embed(decoder_input) # seq_len x batch x decoder_embed_size

encoder_output, hn = encoder(encoder_embeddings) # encoder_output: seq_len x batch_size x enc_hidden_size
                                                 # hn: 1 x batch x enc_hidden_size
hn = hn.squeeze(dim=0)
print('Encoder output:', encoder_output.squeeze(dim=1))


def attention(query, keys, values):
    num_vectors, batch_size, enc_hidden_size = keys.size()
    vector_scores = torch.sum(query.view(1, batch_size, enc_hidden_size) * keys, dim=2) # seq_len x batch_size
    print('Vector scores:', vector_scores.squeeze(dim=1))
    vector_probabilities = F.softmax(vector_scores, dim=0)  # seq_len x batch_size
    print('Vector probab:', vector_probabilities.squeeze(dim=1))
    weighted_vectors = vector_probabilities.view(num_vectors, batch_size, 1) * values # seq_len x batch_size x enc_hidden_size
    context_vectors = torch.sum(weighted_vectors, dim=0) # batch_size x enc_hidden_size
    print('Context vector:',context_vectors.squeeze(dim=0))
    return context_vectors

# Attention t=0
initial_query = torch.zeros((batch_size, encoder_hidden_size))  # 1 x 2 (dec_hidden_size)
context_vectors = attention(initial_query, encoder_output, encoder_output)
decoder_input = torch.cat([decoder_input_embeddings[0], context_vectors], dim=1)
print('Decoder input:', decoder_input)
decoder_output = decoder_cell(decoder_input, hn) # batch_size x dec_hidden_size
print('Decoder output:', decoder_output)

# Attention t=1
context_vectors = attention(decoder_output, encoder_output, encoder_output)
decoder_input = torch.cat([decoder_input_embeddings[1], context_vectors], dim=1)
print('Decoder input:', decoder_input)
decoder_output = decoder_cell(decoder_input, hn) # 1 x dec_hidden_size
print('Decoder output:', decoder_output)
