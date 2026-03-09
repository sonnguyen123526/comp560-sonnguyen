import numpy as np
import pickle

def decode_train_lines(data_dir):
    with open(f'{data_dir}/meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    itos = meta['itos']
    for split in ('train', 'val'):
        tokens = np.fromfile(f'{data_dir}/{split}.bin', dtype=np.uint16)
        text   = ''.join(itos[i] for i in tokens)
        for line in text.split('\n'):
            if line.strip():
                yield line.strip()


def load_test_file(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


datasets = ['data/mixed', 'data/reverse_shared', 'data/addition_shared']

all_train = set()
for d in datasets:
    for line in decode_train_lines(d):
        all_train.add(line)

print(f'Total unique lines across all 3 datasets (train+val): {len(all_train):,}')
print()

rev_test = load_test_file('data/test/reverse_test.txt')
add_test = load_test_file('data/test/addition_test.txt')

rev_overlap = [s for s in rev_test if s in all_train]
add_overlap = [s for s in add_test if s in all_train]

print(f'Reverse test set ({len(rev_test)} samples)')
print(f'  Overlap with training data: {len(rev_overlap)} / {len(rev_test)}  ({100*len(rev_overlap)/len(rev_test):.1f}%)')
if rev_overlap:
    for s in rev_overlap[:3]:
        print(f'    {s}')

print()
print(f'Addition test set ({len(add_test)} samples)')
print(f'  Overlap with training data: {len(add_overlap)} / {len(add_test)}  ({100*len(add_overlap)/len(add_test):.1f}%)')
if add_overlap:
    for s in add_overlap[:3]:
        print(f'    {s}')

print()
if rev_overlap or add_overlap:
    print('WARNING: test/train overlap detected.')
else:
    print('OK: zero overlap between training and test sets.')
