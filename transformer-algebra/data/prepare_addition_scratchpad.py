# Prepare scratchpad addition dataset.
# Train on 2-digit numbers (10-99), test on 3-digit numbers (100-999).
#
# Format: "2 3 + 4 5 -> C:0 8 C:0 6 -> 6 8"

import os
import random
import pickle
import numpy as np

SEED = 42
random.seed(SEED)

TRAIN_FILE = 'data/addition_scratchpad/train.txt'
VAL_FILE   = 'data/addition_scratchpad/val.txt'
TEST_FILE  = 'data/addition_scratchpad/test.txt'


def make_scratchpad(a, b):
    a_digits = [int(d) for d in str(a)]
    b_digits = [int(d) for d in str(b)]

    n = max(len(a_digits), len(b_digits))
    a_digits = [0] * (n - len(a_digits)) + a_digits
    b_digits = [0] * (n - len(b_digits)) + b_digits

    a_lsb = list(reversed(a_digits))
    b_lsb = list(reversed(b_digits))

    carry = 0
    steps = []
    result_digits = []
    for i in range(n):
        s = a_lsb[i] + b_lsb[i] + carry
        carry = s // 10
        digit = s % 10
        steps.append(f"C:{carry} {digit}")
        result_digits.append(digit)

    if carry:
        result_digits.append(carry)
        steps.append(f"C:0 {carry}")

    a_str      = ' '.join(str(d) for d in a_digits)
    b_str      = ' '.join(str(d) for d in b_digits)
    scratch    = ' '.join(steps)
    result_str = ' '.join(str(d) for d in reversed(result_digits))

    return f"{a_str} + {b_str} -> {scratch} -> {result_str}"


def generate_dataset(lo, hi, n, exclude=None):
    exclude = exclude or set()
    samples = []
    attempts = 0
    while len(samples) < n and attempts < n * 20:
        a = random.randint(lo, hi)
        b = random.randint(lo, hi)
        attempts += 1
        if (a, b) in exclude:
            continue
        samples.append((a, b))
        exclude.add((a, b))
    return samples


def save_txt(samples, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for a, b in samples:
            f.write(make_scratchpad(a, b) + '\n')
    print(f"  {len(samples):,} examples -> {path}")
    print(f"  e.g. {make_scratchpad(samples[0][0], samples[0][1])}")


def main():
    os.makedirs('data/addition_scratchpad', exist_ok=True)

    print("Generating 1 and 2 digit training set (1-99)...")
    all_pairs = [(a, b) for a in range(1, 100) for b in range(1, 100)]
    random.shuffle(all_pairs)
    split = int(len(all_pairs) * 0.9)
    train_samples = all_pairs[:split]
    val_samples   = all_pairs[split:]
    save_txt(train_samples, TRAIN_FILE)
    save_txt(val_samples,   VAL_FILE)

    print("\nGenerating 3-digit test set (100-999) OOD...")
    test_samples = generate_dataset(100, 999, 5000)
    save_txt(test_samples, TEST_FILE)

    print("\nTokenizing...")
    with open(TRAIN_FILE) as f:
        train_text = f.read()
    with open(VAL_FILE) as f:
        val_text = f.read()

    chars = sorted(set(train_text))
    stoi  = {c: i for i, c in enumerate(chars)}
    itos  = {i: c for i, c in enumerate(chars)}
    meta  = {'vocab_size': len(chars), 'stoi': stoi, 'itos': itos, 'chars': chars}

    train_ids = np.array([stoi[c] for c in train_text], dtype=np.uint16)
    val_ids   = np.array([stoi[c] for c in val_text],   dtype=np.uint16)

    train_ids.tofile('data/addition_scratchpad/train.bin')
    val_ids.tofile('data/addition_scratchpad/val.bin')

    with open('data/addition_scratchpad/meta.pkl', 'wb') as f:
        pickle.dump(meta, f)

    print(f"\nvocab size  : {len(chars)}")
    print(f"vocab chars : {''.join(chars)!r}")
    print(f"train tokens: {len(train_ids):,}")
    print(f"val tokens  : {len(val_ids):,}")
    print("\nSample traces:")
    for a, b in [(23, 45), (99, 99), (10, 11)]:
        print(f"  {make_scratchpad(a, b)}")


if __name__ == '__main__':
    main()
