import torch
from torch import nn
from tqdm import tqdm

from args import args
from helper import compute_accuracy
from reviewclassifier import ReviewClassifier
from reviewdataset import ReviewDataset, get_num_batches, generate_batches

test_dataset = ReviewDataset.load_dataset_and_load_vectorizer(args.reviews_json_filepath, args.vectorizer_filepath,
                                                    reviews_limit=(args.reviews_train, args.reviews_train+args.reviews_test))

vectorizer = test_dataset.get_vectorizer()
classifier = ReviewClassifier(in_features=len(vectorizer.review_vocab), out_features=len(vectorizer.rating_vocab),
                              hidden_features=args.hidden_features)
classifier.to(args.device)
classifier.load_state_dict(torch.load(args.model_state_filepath, map_location=torch.device(args.device)))
classifier.eval()

loss_func = nn.CrossEntropyLoss()
running_loss = 0.
running_acc = 0.

with tqdm(total=get_num_batches(test_dataset, args.batch_size)) as train_bar:
    batch_generator = generate_batches(test_dataset, batch_size=args.batch_size, device=args.device)

    for batch_index, batch_dict in enumerate(batch_generator):
        # compute the output
        y_pred = classifier(batch_dict['x_data'])

        # compute the loss
        loss = loss_func(y_pred, batch_dict['y_target'])
        loss_t = loss.item()
        running_loss += (loss_t - running_loss) / (batch_index + 1)

        # compute the accuracy
        acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
        running_acc += (acc_t - running_acc) / (batch_index + 1)

        # update bar
        train_bar.set_postfix(loss=running_loss, acc=running_acc)
        train_bar.update()

print(f"Test loss: {running_loss}. Test acc:{running_acc}")