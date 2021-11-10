import numpy as np
import torch
import matplotlib.pyplot as plt

from args import args
from models import NMTModel
from datasets import NMTDataset, get_num_batches, generate_rnn_batches

dataset = NMTDataset.load_dataset_and_load_vectorizer(args.train_filepath, args.vectorizer_filepath)
vectorizer = dataset.get_vectorizer()

model = NMTModel(source_vocab_size=len(vectorizer.source_vocab),
                 source_embedding_size=args.embedding_dim,
                 source_padding_idx=vectorizer.source_vocab[vectorizer.PAD],
                 target_vocab_size=len(vectorizer.target_vocab),
                 target_embedding_size=args.embedding_dim,
                 target_padding_idx=vectorizer.target_vocab[vectorizer.PAD],
                 encoding_size=args.hidden_features,
                 max_target_size=args.max_words,
                 sos_index=vectorizer.target_vocab[vectorizer.SOS])
model.to(args.device)
model.load_state_dict(torch.load(args.model_state_filepath, map_location=torch.device(args.device)))
model.eval()


def plot_attention(translation, ax):
    source_text, target_text, attention_matrix = translation
    source_words = source_text.split()
    target_words = target_text.split()
    source_words = ['<sos>'] + source_words + ['<eos>']
    source_len = len(source_words)
    target_len = len(target_words)
    attention_matrix = attention_matrix[:target_len, :source_len].cpu().numpy().transpose()

    im = ax.imshow(attention_matrix)

    # We want to show all ticks...
    ax.set_xticks(np.arange(target_len))
    ax.set_yticks(np.arange(source_len))
    # ... and label them with the respective list entries
    ax.set_xticklabels(target_words)
    ax.set_yticklabels(source_words)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(source_len):
        for j in range(target_len):
            label = f'{attention_matrix[i, j]:.1f}'
            ax.text(j, i, label, ha="center", va="center", color="w")


def translate(batch_dict):
    y_pred, attention_matrixes = model(batch_dict['x_source'], batch_dict['x_source_length'], None)
    translations = []
    for source, target, attention_matrix in zip(batch_dict['x_source'], y_pred, attention_matrixes):
        source_words = []
        for ind in source[1:]:  # skip <sos>
            word = vectorizer.source_vocab.itos[ind]
            if word == vectorizer.EOS:
                break
            source_words.append(word)
        source_text = ' '.join(source_words)

        target_words = []
        for prob in target:
            ind = torch.argmax(prob).item()
            word = vectorizer.target_vocab.itos[ind]
            if word == vectorizer.EOS or word == vectorizer.PAD:
                break
            target_words.append(word)
        target_text = ' '.join(target_words)

        translations.append((source_text, target_text, attention_matrix))
    return translations

test_batch_size = 9
batch_generator = generate_rnn_batches(dataset, batch_size=test_batch_size, device=args.device)
batch_dict = next(batch_generator)
translations = translate(batch_dict)
fig, axs = plt.subplots(3,3)
for i, translation in enumerate(translations):
    print('> ', translation[0])
    print('< ', translation[1])
    plot_attention(translation, axs[i//3, i%3])

# Rotate the tick labels and set their alignment.

plt.show()

print()
# read data from user
while True:
    sentence = input('> ')
    if not sentence:
        break
    source_vector, source_vector_length = vectorizer.vectorize_source(sentence)
    x_source = source_vector.unsqueeze(0).to(args.device)
    x_source_length = torch.tensor([source_vector_length], dtype=torch.long).to(args.device)
    batch_dict = {'x_source': x_source, 'x_source_length': x_source_length}
    translations = translate(batch_dict)
    for translation in translations:
        print('> ', translation[0])
        print('< ', translation[1])



