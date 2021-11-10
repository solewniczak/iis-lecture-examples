from argparse import Namespace
import torch

args = Namespace(
    lang='pl',

    # Dataset parameters
    max_words=10,
    en_prefixes=(
        'i am ', 'i m ',
        'he is', 'he s ',
        'she is', 'she s ',
        'you are', 'you re ',
        'we are', 'we re ',
        'they are', 'they re '
    ),
    use_prefixes=False,

    # Model parameters
    hidden_features=16, # decrease - because of attention
    embedding_dim=24,

    # Training parameters
    batch_size=32,
    learning_rate=0.01,
    num_epochs=10,
    seed=1337,
    train_split=0.8,  # train/val split

    # Runtime parameters
    cuda=True,
)

args.model_state_filepath = f'nosync/model-{args.lang}.pth'
args.training_history_filepath = f'nosync/traing_history-{args.lang}.json'
args.dataset_filepath = f'nosync/{args.lang}.txt'
args.train_filepath = f'nosync/{args.lang}_train.txt'
args.vectorizer_filepath = f'nosync/vectorizer-{args.lang}.json'

# Check CUDA
if not torch.cuda.is_available():
    args.cuda = False

if args.cuda:
    args.device = 'cuda'
else:
    args.device = 'cpu'

torch.manual_seed(args.seed)