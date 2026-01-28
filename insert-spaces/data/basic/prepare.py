"""
Prepare dataset for inserting spaces between characters.
Uses a clearer prompt format to help the model understand the translation task.
"""
import os
import pickle
import numpy as np
import random

random.seed(42)
alphabet = 'abcdefghijklmnopqrstuvwxyz'

unique_words = set()

# Generate diverse training examples
for _ in range(3000):
    word_len = random.randint(3, 8)
    word = ''.join(random.choice(alphabet) for _ in range(word_len))
    unique_words.add(word)

print(f"Total unique words: {len(unique_words)}")

words = sorted(list(unique_words))

# Build training data with clear input/output separation
blocks = []
target_length = 1_000_000
total_length = 0

while total_length < target_length:
    for word in words:
        spaced = ' '.join(word)
        # Use a clearer format: "Input: word Output: w o r d"
        block = f"Input: {word} Output: {spaced}\n"
        blocks.append(block)
        total_length += len(block)
        if total_length >= target_length:
            break

data = ''.join(blocks)

print("First 10 lines of data:")
lines = data.split('\n')
for i in range(min(10, len(lines))):
    print(lines[i])

print(f"\nlength of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print(f"all the unique characters: {''.join(chars)}")
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# For this experimental example, training and validation data are the same
train_data = data
val_data = data

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
