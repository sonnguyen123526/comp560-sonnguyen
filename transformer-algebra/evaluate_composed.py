import argparse
import os
import pickle
import random
import sys

import torch

sys.path.insert(0, '/Users/sonnguyen/comp560-nanoGPT')
from model import GPT, GPTConfig

SEED = 42

REVERSE_CKPT  = 'out/reverse/ckpt.pt'
ADDITION_CKPT = 'out/addition/ckpt.pt'
COMPOSED_CKPT = 'out/composed/ckpt.pt'


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg  = GPTConfig(**ckpt['model_args'])
    m    = GPT(cfg)
    sd   = {k.removeprefix('_orig_mod.'): v for k, v in ckpt['model'].items()}
    m.load_state_dict(sd)
    m.eval().to(device)
    return m


def load_meta(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


@torch.no_grad()
def greedy(model, prompt_ids, max_new, device):
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    bs = model.config.block_size
    for _ in range(max_new):
        logits, _ = model(x[:, -bs:])
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        x = torch.cat([x, nxt], dim=1)
    return x[0, len(prompt_ids):].tolist()


def reverse_number(model, meta, spaced, device):
    stoi, itos = meta['stoi'], meta['itos']
    n = len(spaced.split())
    prompt = spaced + ' ->'
    try:
        ids = [stoi[c] for c in prompt]
    except KeyError:
        return ''
    out = greedy(model, ids, max_new=n * 2 + 3, device=device)
    raw = ''.join(itos[i] for i in out)
    tokens = raw.split('\n')[0].strip().split()[:n]
    return ' '.join(tokens)


def add_numbers(model, meta, a_spaced, b_spaced, device):
    stoi, itos = meta['stoi'], meta['itos']
    prompt = a_spaced + ' + ' + b_spaced + ' ->'
    try:
        ids = [stoi[c] for c in prompt]
    except KeyError:
        return ''
    out = greedy(model, ids, max_new=12, device=device)
    raw = ''.join(itos[i] for i in out)
    return raw.split('\n')[0].strip()


def make_samples(n):
    random.seed(SEED)
    samples, seen = [], set()
    while len(samples) < n:
        a = random.randint(100, 999)
        b = random.randint(100, 999)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        a_rev = int(str(a)[::-1])
        b_rev = int(str(b)[::-1])
        samples.append((a, b, str(a_rev + b_rev)))
    return samples


def evaluate_pipeline(n, device):
    print("loading reverse model...")
    rev_model = load_model(REVERSE_CKPT,  device)
    rev_meta  = load_meta('data/reverse/meta.pkl')

    print("loading addition model...")
    add_model = load_model(ADDITION_CKPT, device)
    add_meta  = load_meta('data/addition/meta.pkl')

    samples = make_samples(n)
    print(f"\nEvaluating {n} samples — zero-shot pipeline\n")

    total = correct = step1_correct = 0
    wrong = []

    for a, b, expected in samples:
        total += 1
        a_str = ' '.join(str(a))
        b_str = ' '.join(str(b))

        # Step 1: reverse each number
        a_rev = reverse_number(rev_model, rev_meta, a_str, device)
        b_rev = reverse_number(rev_model, rev_meta, b_str, device)

        a_rev_gold = ' '.join(str(a)[::-1])
        b_rev_gold = ' '.join(str(b)[::-1])
        step1_ok   = (a_rev == a_rev_gold and b_rev == b_rev_gold)
        if step1_ok:
            step1_correct += 1

        # Step 2: add reversed numbers
        pred_spaced = add_numbers(add_model, add_meta, a_rev, b_rev, device)
        pred        = ''.join(pred_spaced.split())

        if pred == expected:
            correct += 1
        else:
            wrong.append((a, b, a_rev, b_rev, expected, pred, step1_ok))

    print("=" * 60)
    print("  Strategy : Zero-shot pipeline (no training)")
    print("  Task     : given A+B, compute reverse(A) + reverse(B)")
    print("-" * 60)
    print(f"  Samples          : {total}")
    print(f"  Step-1 accuracy  : {step1_correct}/{total}  ({step1_correct/total*100:.2f}%)  — reversal")
    print(f"  End-to-end acc   : {correct}/{total}  ({correct/total*100:.2f}%)  — full pipeline")
    print("=" * 60)

    if wrong:
        print(f"\n  First {min(10, len(wrong))} wrong predictions:")
        for a, b, a_rev, b_rev, exp, got, s1 in wrong[:10]:
            step1 = 'ok' if s1 else 'fail'
            print(f"    {a}+{b}  step1={step1}  rev=({a_rev},{b_rev})  expected={exp}  got={got!r}")


def evaluate_baseline(n, device):
    if not os.path.exists(COMPOSED_CKPT):
        print(f"No composed checkpoint found at {COMPOSED_CKPT}.")
        print("Train it first:  python train.py config/train_composed.py")
        return

    print("loading composed model...")
    model = load_model(COMPOSED_CKPT, device)
    meta  = load_meta('data/composed/meta.pkl')
    stoi, itos = meta['stoi'], meta['itos']

    samples = make_samples(n)
    print(f"\nEvaluating {n} samples — end-to-end baseline\n")

    total = correct = 0
    wrong = []

    for a, b, expected in samples:
        total += 1
        a_str  = ' '.join(str(a))
        b_str  = ' '.join(str(b))
        prompt = a_str + ' + ' + b_str + ' ->'

        try:
            ids = [stoi[c] for c in prompt]
        except KeyError:
            continue

        out_ids = greedy(model, ids, max_new=25, device=device)
        raw     = ''.join(itos[i] for i in out_ids)

        if '->' in raw:
            result = raw.split('->')[-1].strip().split('\n')[0].strip()
        else:
            result = raw.strip().split('\n')[0].strip()
        pred = ''.join(result.split())

        if pred == expected:
            correct += 1
        else:
            wrong.append((prompt, expected, pred))

    print("=" * 60)
    print("  Strategy : End-to-end baseline (trained on composed data)")
    print("  Task     : given A+B, compute reverse(A) + reverse(B)")
    print("-" * 60)
    print(f"  Samples  : {total}")
    print(f"  Correct  : {correct}")
    print(f"  Accuracy : {correct/total*100:.2f}%")
    print("=" * 60)

    if wrong:
        print(f"\n  First {min(10, len(wrong))} wrong predictions:")
        for prompt, exp, got in wrong[:10]:
            print(f"    {prompt!r}  expected={exp!r}  got={got!r}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', choices=['pipeline', 'baseline', 'all'],
                        default='pipeline')
    parser.add_argument('--n',      type=int,  default=200,
                        help='Number of test samples')
    parser.add_argument('--device', type=str,  default='mps')
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if args.strategy in ('pipeline', 'all'):
        evaluate_pipeline(args.n, args.device)

    if args.strategy in ('baseline', 'all'):
        print()
        evaluate_baseline(args.n, args.device)
