# Evaluate the scratchpad addition model on OOD 3-digit numbers.
# Trained on 2-digit (10-99), tested on 3-digit (100-999).
#
# Usage:
#   python evaluate_addition_scratchpad.py
#   python evaluate_addition_scratchpad.py --n 500 --device mps

import argparse
import os
import pickle
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'comp560-nanoGPT'))
from model import GPT, GPTConfig

CKPT      = 'out/addition_scratchpad/ckpt.pt'
TEST_FILE = 'data/addition_scratchpad/test.txt'
META_FILE = 'data/addition_scratchpad/meta.pkl'
SEED      = 42


def load_model(device):
    ckpt = torch.load(CKPT, map_location=device)
    model = GPT(GPTConfig(**ckpt['model_args']))
    sd = {k.removeprefix('_orig_mod.'): v for k, v in ckpt['model'].items()}
    model.load_state_dict(sd)
    model.eval()
    model.to(device)
    return model, ckpt['iter_num'], float(ckpt['best_val_loss'])


def extract_result(prompt, generated):
    """Pull the answer after the SECOND '->' in the full sequence.

    Format: "A + B -> <scratchpad> -> <result>\n..."
    The prompt already contains the first '->'.
    The model generates '<scratchpad> -> <result>\n...'.
    So we look for the first '->' inside the generated text only.
    """
    # take only the first line of generated output
    first_line = generated.split('\n')[0]
    if '->' not in first_line:
        return ''
    result = first_line.split('->')[-1].strip()
    return ''.join(result.split())


@torch.no_grad()
def greedy_batch(model, all_prompt_ids, max_new, device, batch_size=128):
    pad_id = 0
    results = []
    for start in range(0, len(all_prompt_ids), batch_size):
        batch   = all_prompt_ids[start:start + batch_size]
        max_len = max(len(p) for p in batch)
        padded  = [[pad_id] * (max_len - len(p)) + p for p in batch]
        x = torch.tensor(padded, dtype=torch.long, device=device)
        prompt_lens = [len(p) for p in batch]
        for _ in range(max_new):
            logits, _ = model(x)
            next_ids = logits[:, -1].argmax(dim=-1, keepdim=True)
            x = torch.cat([x, next_ids], dim=1)
        for i, plen in enumerate(prompt_lens):
            results.append(x[i, max_len:].tolist())
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n',      type=int, default=None,  help='number of test samples (default: all)')
    ap.add_argument('--device', type=str, default='mps')
    args = ap.parse_args()

    with open(META_FILE, 'rb') as f:
        meta = pickle.load(f)
    stoi = meta['stoi']
    itos = meta['itos']
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda ids: ''.join(itos.get(i, '?') for i in ids)

    model, itr, val_loss = load_model(args.device)
    print(f"Checkpoint: iter={itr}, best_val_loss={val_loss:.4f}")

    with open(TEST_FILE) as f:
        lines = [l.strip() for l in f if l.strip()]

    if args.n:
        random.seed(SEED)
        lines = random.sample(lines, min(args.n, len(lines)))

    # prompt = everything up to and including the first '->'
    prompts   = [line.split('->')[0].strip() + ' ->' for line in lines]
    expecteds = [''.join(line.split('->')[-1].strip().split()) for line in lines]
    all_ids   = [encode(p) for p in prompts]

    # scratchpad output can be long: 2 steps per digit + result
    max_new = 80

    print(f"Evaluating {len(lines):,} OOD samples (3-digit, trained on 2-digit)...")
    generated = greedy_batch(model, all_ids, max_new, args.device)

    correct = 0
    wrong   = []
    for prompt, expected, gen_ids in zip(prompts, expecteds, generated):
        out_text = decode(gen_ids)
        got = extract_result(prompt, out_text)
        if got == expected:
            correct += 1
        elif len(wrong) < 10:
            wrong.append((prompt, expected, got, out_text.split('\n')[0]))
    total = len(lines)
    print(f"\n{'='*55}")
    print(f"  Addition scratchpad — OOD test accuracy")
    print(f"{'='*55}")
    print(f"  Trained on : 2-digit numbers (10–99)")
    print(f"  Tested on  : 3-digit numbers (100–999)")
    print(f"  Samples    : {total:,}")
    print(f"  Correct    : {correct:,}")
    print(f"  Accuracy   : {correct/total:.2%}")
    print(f"{'='*55}")
    if wrong:
        print(f"\n  First wrong predictions:")
        for prompt, exp, got, raw in wrong:
            print(f"    {prompt}  expected={exp!r}  got={got!r}")
            print(f"    raw: {raw!r}")


if __name__ == '__main__':
    main()
