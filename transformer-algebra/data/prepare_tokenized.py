# Generate and tokenize datasets for transformer composition experiments.
#
# Three tasks:
#   reverse:  "1 2 3 4" -> "4 3 2 1"
#   addition: "1 2 3 + 4 5 6" -> "5 7 9"
#   composed: "1 2 3 + 4 5 6" -> "3 2 1 + 6 5 4" -> "9 7 5"
import os
import pickle
import numpy as np
import random
import argparse

def generate_reverse_dataset(output_file, num_samples=10000, max_len=4):
    samples = set()
    while len(samples) < num_samples:
        length = random.randint(2, max_len)
        num = ''.join(str(random.randint(0, 9)) for _ in range(length))
        sample = f"{' '.join(num)} -> {' '.join(num[::-1])}\n"
        samples.add(sample)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    samples_list = list(samples)
    random.shuffle(samples_list)
    with open(output_file, 'w') as f:
        f.writelines(samples_list)

    print(f"reverse: {len(samples):,} samples -> {output_file}")
    print(f"  e.g. {samples_list[0].strip()}")

def generate_addition_dataset(output_file, num_samples=10000, max_digits=3):
    samples = set()
    max_num = 10 ** max_digits - 1
    min_num = 10 ** (max_digits - 1)

    while len(samples) < num_samples:
        a = random.randint(min_num, max_num)
        b = random.randint(min_num, max_num)
        sample = f"{' '.join(str(a))} + {' '.join(str(b))} -> {' '.join(str(a + b))}\n"
        samples.add(sample)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    samples_list = list(samples)
    random.shuffle(samples_list)
    with open(output_file, 'w') as f:
        f.writelines(samples_list)

    print(f"addition: {len(samples):,} samples -> {output_file}")
    print(f"  e.g. {samples_list[0].strip()}")

def generate_composed_dataset(output_file, num_samples=10000, max_digits=3):
    # format: "1 2 3 + 4 5 6 -> 3 2 1 + 6 5 4 -> 9 7 5"
    samples = set()
    max_num = 10 ** max_digits - 1
    min_num = 10 ** (max_digits - 1)

    while len(samples) < num_samples:
        a = random.randint(min_num, max_num)
        b = random.randint(min_num, max_num)
        a_rev = int(str(a)[::-1])
        b_rev = int(str(b)[::-1])
        result = a_rev + b_rev
        sample = (
            f"{' '.join(str(a))} + {' '.join(str(b))} -> "
            f"{' '.join(str(a_rev))} + {' '.join(str(b_rev))} -> "
            f"{' '.join(str(result))}\n"
        )
        samples.add(sample)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    samples_list = list(samples)
    random.shuffle(samples_list)
    with open(output_file, 'w') as f:
        f.writelines(samples_list)

    print(f"composed: {len(samples):,} samples -> {output_file}")
    print(f"  e.g. {samples_list[0].strip()}")

def tokenize_data(input_file, output_dir):
    if not os.path.exists(input_file):
        print(f"tokenize: {input_file} not found, skipping")
        return

    with open(input_file, 'r') as f:
        data = f.read()

    chars = sorted(set(data))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    data_ids = [stoi[c] for c in data]
    n = len(data_ids)

    train_ids = np.array(data_ids[:int(n * 0.9)], dtype=np.uint16)
    val_ids   = np.array(data_ids[int(n * 0.9):],  dtype=np.uint16)

    os.makedirs(output_dir, exist_ok=True)
    train_ids.tofile(os.path.join(output_dir, 'train.bin'))
    val_ids.tofile(os.path.join(output_dir, 'val.bin'))

    meta = {'vocab_size': len(chars), 'itos': itos, 'stoi': stoi}
    with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    print(f"tokenized {output_dir}: {len(train_ids):,} train / {len(val_ids):,} val tokens  (vocab={''.join(chars)!r})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--max_digits',  type=int, default=3)
    parser.add_argument('--max_len',     type=int, default=4,
                        help='max digit length for reverse task (5-digit held out for test)')
    parser.add_argument('--generate-only', action='store_true')
    parser.add_argument('--tokenize-only', action='store_true')
    args = parser.parse_args()

    random.seed(42)

    datasets = [
        ('data/reverse',  lambda f: generate_reverse_dataset(f,  args.num_samples, args.max_len)),
        ('data/addition', lambda f: generate_addition_dataset(f, args.num_samples, args.max_digits)),
        ('data/composed', lambda f: generate_composed_dataset(f, args.num_samples, args.max_digits)),
    ]

    if not args.tokenize_only:
        for dirname, generator in datasets:
            generator(f'{dirname}/train.txt')

    if not args.generate_only:
        for dirname, _ in datasets:
            tokenize_data(f'{dirname}/train.txt', dirname)

if __name__ == '__main__':
    main()
