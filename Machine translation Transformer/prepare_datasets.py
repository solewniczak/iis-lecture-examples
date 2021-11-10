from args import args
from helper import normalize_string


def prepare_datasets():
    print('reading pairs...')
    separator = '\t'

    lines = open(args.dataset_filepath, encoding='utf-8').read().strip().split('\n')
    pairs = []
    for line in lines:
        cols = line.split(separator)
        lang1 = normalize_string(cols[0])
        lang2 = normalize_string(cols[1])
        lang1_tokens = lang1.split()
        lang2_tokens = lang2.split()
        if len(lang1_tokens) < args.max_words and len(lang2_tokens) < args.max_words:
            if not args.use_prefixes or lang1.startswith(args.en_prefixes):
                pairs.append((lang1, lang2))

    print(f'total pairs: {len(pairs)}')
    with open(args.train_filepath, 'w') as fp:
        for pair in pairs:
            fp.write(separator.join(pair))
            fp.write('\n')