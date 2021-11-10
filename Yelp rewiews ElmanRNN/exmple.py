import torch
import torch.nn as nn

rnn = nn.RNNCell(input_size=3, hidden_size=2, bias=False, nonlinearity='relu')
rnn.weight_ih.data = torch.tensor([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]])
rnn.weight_hh.data = torch.tensor([[0.3, 0.3], [0.4, 0.4]])

input = torch.tensor([[[1.0, 0.9, 0.8]], [[0.7, 0.6, 0.5]]]) # seq x batch_size x input_size
hx = torch.zeros((1,2)) # 1x2
output = []
for i in range(2):
    hx = rnn(input[i], hx)
    output.append(hx)
print(output)

rnn = nn.RNN(input_size=3, hidden_size=2, bias=False, nonlinearity='relu')
rnn.weight_ih_l0.data = torch.tensor([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]])
rnn.weight_hh_l0.data = torch.tensor([[0.3, 0.3], [0.4, 0.4]])

input = torch.tensor([[[1.0, 0.9, 0.8]], [[0.7, 0.6, 0.5]]]) # seq x batch_size x input_size
h0 = torch.zeros((1,1,2)) # 1x1x2
output, hn = rnn(input, h0)
print(output)