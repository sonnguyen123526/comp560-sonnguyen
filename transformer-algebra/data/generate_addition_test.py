# Generate held-out test set for the addition model.
# Excludes pairs seen in train.txt, stratified by carry count.
import os
import random

TRAIN_FILE = os.path.join(os.path.dirname(__file__), 'addition', 'train.txt')
TEST_FILE  = os.path.join(os.path.dirname(__file__), 'addition', 'test.txt')
NUM_TEST   = 5000
SEED       = 123


def count_carries(a, b):
    carries = 0
    carry = 0
    for d in range(max(len(str(a)), len(str(b)))):
        da = (a // 10**d) % 10
        db = (b // 10**d) % 10
        s = da + db + carry
        carry = s // 10
        if carry:
            carries += 1
    return carries


def main():
    random.seed(SEED)

    # load training pairs to exclude from test set
    seen = set()
    with open(TRAIN_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lhs = line.split('->')[0].strip()          # "1 2 3 + 4 5 6"
            parts = lhs.split('+')
            a = int(''.join(parts[0].split()))
            b = int(''.join(parts[1].split()))
            seen.add((a, b))

    buckets = {0: [], 1: [], 2: [], 3: []}
    attempts = 0
    while min(len(v) for v in buckets.values()) < NUM_TEST // 4 and attempts < 10_000_000:
        a = random.randint(100, 999)
        b = random.randint(100, 999)
        attempts += 1
        if (a, b) in seen:
            continue
        c = count_carries(a, b)
        c = min(c, 3)
        if len(buckets[c]) < NUM_TEST // 4:
            buckets[c].append((a, b))

    samples = []
    for c, pairs in buckets.items():
        for a, b in pairs:
            result = a + b
            line = f"{' '.join(str(a))} + {' '.join(str(b))} -> {' '.join(str(result))}"
            samples.append((c, line))

    random.shuffle(samples)

    os.makedirs(os.path.dirname(TEST_FILE), exist_ok=True)
    with open(TEST_FILE, 'w') as f:
        for _, line in samples:
            f.write(line + '\n')

    print(f"Test samples written: {len(samples):,} -> {TEST_FILE}")
    print(f"Carry breakdown:")
    for c in sorted(buckets):
        print(f"  {c} carries: {len(buckets[c]):,}")
    print(f"\nFirst 5 examples:")
    for _, line in samples[:5]:
        print(f"  {line}")


if __name__ == '__main__':
    main()
