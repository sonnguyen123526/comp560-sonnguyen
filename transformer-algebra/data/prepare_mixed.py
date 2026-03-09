import argparse
import os
import pickle
import random

import numpy as np


def all_reverse_examples():
    """Return every digit-reversal example for 2-, 3-, and 4-digit numbers."""
    examples = []
    for length in range(2, 5):
        for i in range(10 ** length):
            num = str(i).zfill(length)
            inp = " ".join(num)
            out = " ".join(num[::-1])
            examples.append(f"{inp} -> {out}\n")
    return examples


def sample_addition_examples(n, exclude, seed):
    """Sample n unique addition examples (a + b, 100 ≤ a,b ≤ 999) not already in exclude."""
    rng = random.Random(seed)
    samples = []
    seen = set(exclude)
    attempts = 0
    while len(samples) < n:
        a = rng.randint(100, 999)
        b = rng.randint(100, 999)
        line = f"{' '.join(str(a))} + {' '.join(str(b))} -> {' '.join(str(a + b))}\n"
        if line not in seen:
            seen.add(line)
            samples.append(line)
        attempts += 1
        if attempts > n * 100:
            raise RuntimeError("Could not sample enough unique addition examples.")
    return samples


def build_vocab(text):
    chars = sorted(set(text))
    stoi  = {ch: i for i, ch in enumerate(chars)}
    itos  = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def tokenize(lines, stoi):
    ids = []
    for line in lines:
        ids.extend(stoi[c] for c in line)
    return np.array(ids, dtype=np.uint16)


def save_dataset(out_dir, train_lines, val_lines, stoi, itos):
    os.makedirs(out_dir, exist_ok=True)
    train_ids = tokenize(train_lines, stoi)
    val_ids   = tokenize(val_lines,   stoi)
    train_ids.tofile(os.path.join(out_dir, "train.bin"))
    val_ids.tofile(os.path.join(out_dir, "val.bin"))
    meta = {"vocab_size": len(stoi), "stoi": stoi, "itos": itos}
    with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    print(f"  {out_dir}: {len(train_ids):,} train / {len(val_ids):,} val tokens"
          f"  ({len(train_lines):,} + {len(val_lines):,} examples)")


def save_test_file(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.writelines(lines)
    print(f"  {path}: {len(lines):,} held-out examples")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Run from the project root regardless of where the script is called from
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    rng = random.Random(args.seed)

    # Reverse: enumerate all 11,100 examples (lengths 2-4), then index-split
    print("Building reverse partition ...")
    all_rev = all_reverse_examples()
    rng.shuffle(all_rev)
    rev_base, rev_ft, rev_test = all_rev[:7000], all_rev[7000:9000], all_rev[9000:9500]
    print(f"  base={len(rev_base):,}  ft={len(rev_ft):,}  test={len(rev_test):,}  unused={len(all_rev)-9500:,}")

    # Addition: sample disjoint subsets from 810,000-pair space
    print("\nBuilding addition partition ...")
    add_base = sample_addition_examples(10000, exclude=set(),                          seed=args.seed + 10)
    add_ft   = sample_addition_examples(5000,  exclude=set(add_base),                 seed=args.seed + 20)
    add_test = sample_addition_examples(1000,  exclude=set(add_base) | set(add_ft),   seed=args.seed + 30)
    print(f"  base={len(add_base):,}  ft={len(add_ft):,}  test={len(add_test):,}")

    # Build a single vocabulary from all splits so every model shares identical architecture
    all_text   = "".join(rev_base + rev_ft + rev_test + add_base + add_ft + add_test)
    stoi, itos = build_vocab(all_text)
    print(f"\nVocab ({len(stoi)} chars): {''.join(sorted(stoi))!r}")

    # Write binary datasets (90/10 train/val split each)
    print()
    mixed = rev_base + add_base
    rng.shuffle(mixed)
    cut = int(len(mixed) * 0.9)
    save_dataset("data/mixed",            mixed[:cut],   mixed[cut:],   stoi, itos)

    rng.shuffle(rev_ft)
    cut = int(len(rev_ft) * 0.9)
    save_dataset("data/reverse_shared",   rev_ft[:cut],  rev_ft[cut:],  stoi, itos)

    rng.shuffle(add_ft)
    cut = int(len(add_ft) * 0.9)
    save_dataset("data/addition_shared",  add_ft[:cut],  add_ft[cut:],  stoi, itos)

    # Save held-out test files (never touched during any training run)
    print()
    save_test_file("data/test/reverse_test.txt",  rev_test)
    save_test_file("data/test/addition_test.txt", add_test)

    # Sanity check: confirm zero leakage between training and test sets
    train_all = set(rev_base) | set(rev_ft) | set(add_base) | set(add_ft)
    test_all  = set(rev_test) | set(add_test)
    leak = train_all & test_all
    if leak:
        print(f"\nERROR: {len(leak)} test examples appear in training data!")
    else:
        print("\nOK: zero overlap between training and test sets.")


if __name__ == "__main__":
    main()
