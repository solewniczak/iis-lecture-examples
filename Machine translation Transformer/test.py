import torch
from torch.nn import functional as F

from args import args
from models import Transformer
from datasets import NMTDataset

dataset = NMTDataset.load_dataset_and_load_vectorizer(args.train_filepath, args.vectorizer_filepath)
vectorizer = dataset.get_vectorizer()

model = Transformer(len(vectorizer.source_vocab),
                    len(vectorizer.target_vocab),
                    args.d_model, args.N, args.heads)
model.to(args.device)
model.load_state_dict(torch.load(args.model_state_filepath, map_location=torch.device(args.device)))
model.eval()


def translate(sentence):
    source_vector = vectorizer.vectorize_source(sentence)
    source_mask = vectorizer.create_masks(source_vector)

    # batch the data
    source_vector = source_vector.unsqueeze(0)
    source_mask = source_mask.unsqueeze(0)

    encoder_outputs = model.encoder(source_vector, source_mask)

    outputs = torch.zeros(args.max_words).type_as(source_vector.data)
    outputs[0] = torch.LongTensor([vectorizer.target_vocab.stoi[vectorizer.SOS]])

    for i in range(1, args.max_words):
        target_mask = vectorizer.nopeak_mask(i).unsqueeze(0)

        decoder_outputs = model.decoder(outputs[:i].unsqueeze(0), encoder_outputs, source_mask, target_mask)
        out = model.out(decoder_outputs)
        out = F.softmax(out, dim=-1)
        val, ix = out[:, -1].data.topk(1)

        outputs[i] = ix[0][0]
        if ix[0][0] == vectorizer.target_vocab.stoi[vectorizer.EOS]:
            break

    return ' '.join(
        [vectorizer.target_vocab.itos[ix] for ix in outputs[1:i]]
    )

test_samples = 10
for i in range(test_samples):
    random_idx = torch.randint(len(dataset), (1,)).item()
    pair = dataset.get_pair(random_idx)
    translation = translate(pair[0])
    print('> ', pair[0])
    print('< ', translation)

# read data from user
while True:
    sentence = input('> ')
    if not sentence:
        break
    translation = translate(sentence)
    print('< ', translation)



