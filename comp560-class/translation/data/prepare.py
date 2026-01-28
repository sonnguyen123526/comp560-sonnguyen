"""
Prepare a simple Vietnamese-English number translation dataset.
Maps character sequences to integers for sequence-to-sequence learning.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import numpy as np
import random

# Vietnamese numbers 0-20
vietnamese_numbers = {
    0: "không", 1: "một", 2: "hai", 3: "ba", 4: "bốn",
    5: "năm", 6: "sáu", 7: "bảy", 8: "tám", 9: "chín",
    10: "mười", 11: "mười một", 12: "mười hai", 13: "mười ba", 14: "mười bốn",
    15: "mười năm", 16: "mười sáu", 17: "mười bảy", 18: "mười tám",
    19: "mười chín", 20: "hai mươi"
}

english_numbers = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
    10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
    15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen",
    19: "nineteen", 20: "twenty"
}

# Generate random translation pairs
random.seed(42)
num_samples = 50000
data_pairs = []

for _ in range(num_samples):
    num = random.randint(0, 20)
    data_pairs.append(f"{vietnamese_numbers[num]} -> {english_numbers[num]}\n")

data = ''.join(data_pairs)
print(f"length of dataset in characters: {len(data):,}")
print(f"number of translation pairs: {len(data_pairs):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers

def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

# Save to data/translation/ subdirectory
output_dir = os.path.join(os.path.dirname(__file__), 'translation')
os.makedirs(output_dir, exist_ok=True)
train_ids.tofile(os.path.join(output_dir, 'train.bin'))
val_ids.tofile(os.path.join(output_dir, 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
