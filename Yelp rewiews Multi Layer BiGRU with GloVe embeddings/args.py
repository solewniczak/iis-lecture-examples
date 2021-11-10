import os
from argparse import Namespace
import torch

args = Namespace(
    # Filepaths
    model_state_filepath='model-data/model.pth',
    training_history_filepath='model-data/traing_history.json',
    reviews_json_filepath=os.path.expanduser('~/iss-lecture-examples/yelp_academic_dataset_review.json'),
    reviews_json_train_filepath='model-data/yelp_train.json',
    reviews_json_test_filepath='model-data/yelp_test.json',
    vectorizer_filepath='model-data/vectorizer.json',

    # Dataset parameters
    words_limit=1000,
    max_review_words=128,
    reviews_train=20000,
    reviews_test=5000,

    # Model parameters
    hidden_features=32,
    embedding_dim=100,
    num_layers=4, # NEW!

    # Training parameters
    batch_size=128,
    learning_rate=0.01,
    num_epochs=10,
    seed=1337,
    dropout_p=0.2, # NEW!

    # Runtime parameters
    cuda=True,
)

# Check CUDA
if not torch.cuda.is_available():
    args.cuda = False

if args.cuda:
    args.device = 'cuda'
else:
    args.device = 'cpu'

torch.manual_seed(args.seed)

# create directories
os.makedirs(os.path.dirname(args.model_state_filepath), exist_ok=True)
os.makedirs(os.path.dirname(args.training_history_filepath), exist_ok=True)
os.makedirs(os.path.dirname(args.vectorizer_filepath), exist_ok=True)
os.makedirs(os.path.dirname(args.reviews_json_train_filepath), exist_ok=True)
os.makedirs(os.path.dirname(args.reviews_json_test_filepath), exist_ok=True)