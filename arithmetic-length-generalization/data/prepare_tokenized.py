"""
Tokenize datasets for nanoGPT training.
Converts text to binary format with character-level tokenization.
"""
import os
import pickle
import numpy as np

def tokenize_dataset(input_file, output_dir):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found")
        return
    
    print(f"Processing {input_file}")
    
    with open(input_file, 'r') as f:
        data = f.read()
    
    chars = sorted(set(data))
    vocab_size = len(chars)
    print(f"  Vocab: {vocab_size} chars")
    if len(chars) < 100:
        print(f"  {repr(''.join(chars))}")
    
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    
    data_ids = encode(data)
    split_idx = int(len(data_ids) * 0.9)
    train_ids = np.array(data_ids[:split_idx], dtype=np.uint16)
    val_ids = np.array(data_ids[split_idx:], dtype=np.uint16)
    
    os.makedirs(output_dir, exist_ok=True)
    train_ids.tofile(os.path.join(output_dir, 'train.bin'))
    val_ids.tofile(os.path.join(output_dir, 'val.bin'))
    
    with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump({'vocab_size': vocab_size, 'itos': itos, 'stoi': stoi}, f)
    
    print(f"  ✓ {len(train_ids):,} train / {len(val_ids):,} val tokens")

def main():
    print("\nTokenizing datasets...")
    
    datasets = [
        ('random_spaces/train.txt', 'random_spaces'),
        ('cyclic_shifts/train.txt', 'cyclic_shifts')
    ]
    
    for input_file, output_dir in datasets:
        print(f"\n{output_dir}:")
        tokenize_dataset(input_file, output_dir)
    
    print("\n✓ Done! Each dataset has train.bin, val.bin, and meta.pkl")
    print("\nTrain with:")
    print("  NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \\")
    print("    python -u ../../comp560-nanoGPT/train.py config/random_spaces.py")

if __name__ == '__main__':
    main()
