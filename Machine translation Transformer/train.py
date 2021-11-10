import json
import os

import torch
from torch import nn, optim
from tqdm import tqdm

from args import args
from helper import count_parameters, sequence_loss, compute_accuracy
from models import Transformer
from datasets import NMTDataset, get_num_batches, generate_batches
from prepare_datasets import prepare_datasets

if os.path.exists(args.model_state_filepath):
    print('Model file already exists. Press enter to override and start new training.')
    input()

if not os.path.exists(args.train_filepath):
    prepare_datasets()

# create dataset and vectorizer
dataset = NMTDataset.load_dataset_and_make_vectorizer(args.train_filepath, args.max_words)
dataset.save_vectorizer(args.vectorizer_filepath)
vectorizer = dataset.get_vectorizer()

model = Transformer(len(vectorizer.source_vocab),
                    len(vectorizer.target_vocab),
                    args.d_model, args.N, args.heads)
model.to(args.device)

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

print(f'The models has {count_parameters(model):,} trainable parameters')

train_size = int(args.train_split * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

training_loss = []
training_acc = []
val_loss = []
val_acc = []
best_acc = 0.0
with tqdm(total=args.num_epochs * get_num_batches(train_dataset, args.batch_size)) as train_bar:
    for epoch_index in range(args.num_epochs):

        batch_generator = generate_batches(train_dataset, batch_size=args.batch_size, device=args.device)

        running_loss = 0.0
        running_acc = 0.0
        model.train()
        for batch_index, batch_dict in enumerate(batch_generator):
            # the training routine is these 5 steps:

            # --------------------------------------
            # step 1. zero the gradients
            optimizer.zero_grad()

            # step 2. compute the output
            y_pred = model(batch_dict['x_source'], batch_dict['x_target'],
                           batch_dict['source_mask'], batch_dict['target_mask'])

            # step 3. compute the loss
            loss = sequence_loss(y_pred, batch_dict['y_target'])
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            # step 4. use loss to produce gradients
            loss.backward()

            # step 5. use optimizer to take gradient step
            optimizer.step()
            # -----------------------------------------
            # compute the accuracy
            acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
            running_acc += (acc_t - running_acc) / (batch_index + 1)

            # update bar
            train_bar.set_postfix(loss=running_loss, acc=running_acc, epoch=epoch_index)
            train_bar.update()

        # save training state
        training_loss.append(running_loss)
        training_acc.append(running_acc)

        # Validation
        batch_generator = generate_batches(val_dataset, batch_size=args.batch_size, device=args.device)
        running_loss = 0.
        running_acc = 0.
        model.eval()

        for batch_index, batch_dict in enumerate(batch_generator):
            # compute the output
            y_pred = model(batch_dict['x_source'], batch_dict['x_target'],
                           batch_dict['source_mask'], batch_dict['target_mask'])

            # compute the loss
            loss = sequence_loss(y_pred, batch_dict['y_target'])
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            # compute the accuracy
            acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
            running_acc += (acc_t - running_acc) / (batch_index + 1)

        print(f'\nEpoch {epoch_index} validation loss:{running_loss}, validation acc:{running_acc}')

        # save validation state
        val_loss.append(running_loss)
        val_acc.append(running_acc)

        with open(args.training_history_filepath, 'w') as fp:
            json.dump({
                'training_loss': training_loss,
                'training_acc': training_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, fp)

        # save first models or if acc is better than best acc
        if len(val_acc) == 1 or running_acc > best_acc:
            best_acc = running_acc
            torch.save(model.state_dict(), args.model_state_filepath)
            print(f'Validation accuracy increased. Saving models: {epoch_index}')