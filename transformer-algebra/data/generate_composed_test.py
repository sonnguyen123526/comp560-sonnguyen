# Generate a held-out test set for the composed task (reverse-then-add).
# Excludes all pairs seen in data/composed/train.txt.

import os
import random

TRAIN_FILE = os.path.join(os.path.dirname(__file__), 'composed', 'train.txt')
TEST_FILE  = os.path.join(os.path.dirname(__file__), 'composed', 'test.txt')
NUM_TEST   = 3000
SEED       = 99


def main():
    random.seed(SEED)

    seen = set()
    with open(TRAIN_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lhs = line.split('->')[0].strip()
            parts = lhs.split('+')
            a = int(''.join(parts[0].split()))
            b = int(''.join(parts[1].split()))
            seen.add((a, b))

    print(f"Training pairs loaded: {len(seen):,}")

    samples = []
    attempts = 0
    while len(samples) < NUM_TEST and attempts < 10_000_000:
        a = random.randint(100, 999)
        b = random.randint(100, 999)
        attempts += 1
        if (a, b) in seen:
            continue
        a_rev = int(str(a)[::-1])
        b_rev = int(str(b)[::-1])
        result = a_rev + b_rev
        line = (
            f"{' '.join(str(a))} + {' '.join(str(b))} -> "
            f"{' '.join(str(a_rev))} + {' '.join(str(b_rev))} -> "
            f"{' '.join(str(result))}"
        )
        samples.append(line)
        seen.add((a, b))

    random.shuffle(samples)
    os.makedirs(os.path.dirname(TEST_FILE), exist_ok=True)
    with open(TEST_FILE, 'w') as f:
        for line in samples:
            f.write(line + '\n')

    print(f"Test samples written: {len(samples):,} -> {TEST_FILE}")
    print(f"First 3 examples:")
    for line in samples[:3]:
        print(f"  {line}")


if __name__ == '__main__':
    main()
