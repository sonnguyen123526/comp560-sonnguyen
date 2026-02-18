# Tokenize datasets for training
# Character-level tokenization - each char gets an ID

import os
import pickle
import numpy as np
import sys

def prepare_data(dataset_name):
    data_dir = dataset_name
    
    # read the training data
    input_file = os.path.join(data_dir, 'train.txt')
    with open(input_file, 'r') as f:
        data = f.read()
    
    print(f"Dataset: {dataset_name}")
    print(f"  Total chars: {len(data):,}")
    
    # get all unique characters
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print(f"  Vocab size: {vocab_size}")
    print(f"  Chars: {''.join(chars)}")
    
    # create mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    def encode(s):
        return [stoi[c] for c in s]
    
    def decode(l):
        return ''.join([itos[i] for i in l])
    
    # encode everything
    encoded_data = np.array(encode(data), dtype=np.uint16)
    
    # train/val split (90/10)
    n = len(encoded_data)
    train_data = encoded_data[:int(n*0.9)]
    val_data = encoded_data[int(n*0.9):]
    
    print(f"  Train: {len(train_data):,}")
    print(f"  Val: {len(val_data):,}")
    
    # save binary files
    train_data.tofile(os.path.join(data_dir, 'train.bin'))
    val_data.tofile(os.path.join(data_dir, 'val.bin'))
    
    # save metadata
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
        'encode': encode,
        'decode': decode,
    }
    
    with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    
    print(f"Saved to {data_dir}/")
    
    # quick test
    sample = data[:100]
    assert sample == decode(encode(sample))
    print("Encoding verified ✓")
    
    print(f"\nSample data:")
    lines = data.split('\n')[:5]
    for line in lines:
        print(f"  {line}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python prepare_tokenized.py <dataset_name>")
        print("Example: python prepare_tokenized.py addition")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    prepare_data(dataset_name)
